import math
import pika
from requests.auth import HTTPBasicAuth
from urllib.parse import quote
import requests
import os
import numpy as np
import cv2


_MB = 1024 * 1024

# One decoded frame exactly as the edge puts it on the wire: 640x640 RGB
# float32. Scheduler.first_layer resizes every frame to 640x640, scales it to
# float32, then clones one contiguous tensor per frame into the message, so a
# raw-frame batch body is batch_size x this, give or take a few hundred bytes
# of pickle framing. Used only to report sizes in the startup banner.
FRAME_BYTES = 640 * 640 * 3 * 4          # 4,915,200 B = 4.6875 MiB

# Broker-side mutex proving one server owns the run — see acquire_server_lock.
SERVER_LOCK_QUEUE = "server_lock"


def cloud_work_queue(base, index):
    """The dedicated raw-frame work queue for one cloud.

    Each cloud consumes ONLY its own queue. That is what makes chunking work:
    with a shared queue the 3 clouds compete, so chunk 1 of a batch goes to one
    cloud and chunk 2 to another and neither can merge them. Giving each cloud
    its own queue means every chunk an edge sends for a batch lands on the same
    consumer by construction, with no claim handshake.

    `base` stays the CLUSTER identity (intermediate_queue, or
    intermediate_queue_k under clustering) and is what the FPS pings, metrics
    filenames, fanout exchange and utilization reports are keyed on — the
    per-cloud suffix must never leak into those, or one cluster's results
    fragment into three.
    """
    return f"{base}_c{int(index)}"


def slot_queue_for(work_queue):
    """Permit pool guarding one cloud work queue. Named with the 'slot_queue'
    prefix so delete_old_queues still deletes it (its arguments change with
    `a`, and re-declaring with different arguments is a PRECONDITION_FAILED)."""
    return f"slot_queue_{work_queue}"


def _load_config(config):
    """Return the given config dict, or read config.yaml when called with None
    (the edge/cloud Scheduler doesn't carry the config object around)."""
    if config is None:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config


def transport_plan(config=None):
    """Turn `a` (rabbit.max-queue-messages) into the queue geometry.

    `a` is the cap on ONE cloud work queue, measured in whole raw-frame batch
    messages, and it may be fractional:

        a = 2     2 full batch messages resident per cloud queue
        a = 1     1 full batch message
        a = 0.5   half a batch: the batch is cut into 2 chunks, 1 resident
        a = 0.25  a quarter:    4 chunks, 1 resident

    a >= 1 keeps one message per batch and caps the queue at floor(a). a < 1
    cannot be expressed as a message count at all, so the MESSAGE is made
    smaller instead: a chunk carries floor(a x batch_size) frames and the queue
    holds one of them, which takes ceil(batch_size / that) chunks to move a
    batch. Flooring the frames per chunk keeps the resident payload at or below
    `a`, so the bound is undershot, never exceeded — with one exception: a frame
    is indivisible, so for a < 1/batch_size the chunk stays at one frame and the
    resident payload stops shrinking. The banner prints the payload it actually
    settled on.

    Chunks are labelled in the AMQP headers (batch_key / chunk_index /
    chunk_total) and merged by the cloud before inference, so a batch stays one
    unit of work no matter how many messages carried it: latency, e2e, FPS and
    mAP accounting are all untouched by `a`.

    Pure calculation — never raises. `capacity` is 0 when `a` is unset, which
    means no cap and no permits, i.e. unbounded broker memory.

    Every machine must compute the SAME numbers: `capacity` becomes the queue's
    x-max-length and a mismatched queue_declare is a PRECONDITION_FAILED. So
    `a` and server.batch-size must be identical in every config.yaml.
    """
    config = _load_config(config)
    rabbit = config.get('rabbit', {}) or {}
    server = config.get('server') or {}
    batch_size = int(server.get('batch-size') or 0)
    clients = server.get('clients') or []
    num_clouds = int(clients[-1]) if len(clients) > 1 else 1
    a = float(rabbit.get('max-queue-messages') or 0)

    if a <= 0 or batch_size <= 0:
        capacity, frames_per_chunk = 0, batch_size
    elif a >= 1:
        capacity, frames_per_chunk = int(a), batch_size
    else:
        # Size the CHUNK from `a` (floor), then count how many it takes — not the
        # other way round. Deriving frames_per_chunk as ceil(batch/chunks) rounds
        # the payload UP and can overshoot: a=0.1 of 32 frames wants 3.2 frames,
        # and ceil(32/10)=4 frames is 25% over the bound. One frame is the floor;
        # below that (a < 1/batch_size) the chunk cannot shrink further and the
        # resident payload is one frame regardless of `a`.
        capacity = 1
        frames_per_chunk = max(1, int(math.floor(a * batch_size)))

    chunks = int(math.ceil(batch_size / frames_per_chunk)) if frames_per_chunk else 1
    chunk_bytes = frames_per_chunk * FRAME_BYTES
    return {
        "a": a,
        "batch_size": batch_size,
        "num_clouds": num_clouds,
        "chunks": chunks,                        # messages per batch
        "frames_per_chunk": frames_per_chunk,
        "capacity": capacity,                    # x-max-length AND permits per pool
        "body_bytes": batch_size * FRAME_BYTES,  # one whole batch
        "chunk_bytes": chunk_bytes,              # one message
        "peak_per_queue_bytes": capacity * chunk_bytes,
        "peak_total_bytes": capacity * chunk_bytes * num_clouds,
    }


def describe_transport_plan(plan):
    """Startup banner spelling out the geometry `a` produced, so a run's log
    records the bound it actually operated under. Reports the whole-broker total
    as well as the per-queue cap: `a` caps ONE cloud queue, and there is one
    such queue per cloud, so the broker holds up to num_clouds x that."""
    if plan["capacity"] <= 0:
        return ("[Slots] disabled — rabbit.max-queue-messages is unset; queues are "
                "uncapped, no publish permits, broker memory is UNBOUNDED")
    split = (f"batch of {plan['batch_size']} split into {plan['chunks']} x "
             f"{plan['frames_per_chunk']} frames"
             if plan["chunks"] > 1 else
             f"batch of {plan['batch_size']} in one message")
    return (
        f"[Slots] a={plan['a']:g} -> {split}, {plan['capacity']} resident per cloud "
        f"queue = {plan['peak_per_queue_bytes'] / _MB:.0f} MiB each; "
        f"{plan['num_clouds']} cloud queues -> {plan['peak_total_bytes'] / _MB:.0f} MiB "
        f"peak in the broker"
    )


def _overflow_args(max_len, overflow):
    """Build a RabbitMQ arguments dict that enforces broker-level overflow:
    ``{'x-max-length': N, 'x-overflow': 'reject-publish'}``. The broker bounds
    the queue at N messages and rejects (NACKs) further publishes once full,
    instead of buffering until it runs out of RAM. Returns ``None`` (no limit)
    when max_len is unset/0, keeping queue_declare backward compatible.

    NOTE: every queue_declare for the *same* queue MUST pass identical arguments,
    or RabbitMQ raises PRECONDITION_FAILED and closes the channel. So a given
    queue must always be declared via the same getter below."""
    if not max_len:
        return None
    return {
        'x-max-length': int(max_len),
        'x-overflow': overflow,
    }


def get_intermediate_queue_args(config=None):
    """Overflow args for a raw-frame work queue — `capacity` messages, which is
    also the size of that queue's permit pool, so a held permit always
    corresponds to a free slot and a permitted publish can never be NACKed.

    Also used for the shared base intermediate_queue, which in only_edge / split
    mode carries whole (unchunked) messages."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    return _overflow_args(transport_plan(config)["capacity"],
                          rabbit.get('overflow', 'reject-publish'))


def get_bbox_queue_args(config=None):
    """Overflow args for bbox_queue — the light queue carrying edge-computed
    bboxes (~KB each, text). Capped by `bbox-max-queue-messages`, which can be
    much deeper than the image queue for the same RAM. Falls back to the
    raw-frame permit count if the bbox-specific key is unset."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    max_len = rabbit.get('bbox-max-queue-messages') or transport_plan(config)["capacity"]
    return _overflow_args(max_len, rabbit.get('overflow', 'reject-publish'))


def get_slot_queue_args(config=None):
    """Args for a permit pool (one per cloud work queue, slot_queue_for()).

    The cap is the whole point: it makes the pool physically incapable of
    holding more permits than `a` allows, so broker RAM stays bounded even when
    permit accounting goes wrong. It can go wrong easily — the cloud mints a
    permit for every chunk it pulls, so ANY message that was not paid for (a
    leftover from a previous run, a batch from an edge still running older code
    during a rolling restart) inflates the pool. An inflated pool hands out
    permits instantly, every edge routes to the cloud, and all 9 push ~150 MB
    at once. Without this cap that failure is unbounded.

    'drop-head' rather than 'reject-publish': permits are fungible, so when the
    pool is full the right thing is to quietly discard the surplus one.
    """
    max_len = get_publish_slots(config)
    if not max_len:
        return None
    return {'x-max-length': int(max_len), 'x-overflow': 'drop-head'}


def get_publish_slots(config=None):
    """Permits in ONE cloud queue's pool — how many chunks that cloud may have
    resident at once. Equal to that queue's x-max-length deliberately: an edge
    takes a permit from the pool BEFORE it transmits, so holding a permit
    guarantees a free slot and the publish cannot be NACKed. That matters
    because a NACK costs a full retransmission of the body.

    'x-max-length' alone cannot bound broker memory — the broker must receive a
    message in full before it can evaluate the limit and reject it, so all 9
    edges can each push a complete body before any one is refused. The permit is
    taken up front, so the resident count is bounded by the permit count rather
    than by the edge count.

    Returns 0 when `a` is unset, which means uncapped and unpermitted.
    """
    return transport_plan(config)["capacity"]


def acquire_server_lock(address, username, password, virtual_host):
    """Broker-side mutex that makes a second concurrent server impossible.

    `queue_declare(exclusive=True)` binds the queue to THIS connection: any
    other connection declaring the same name gets 405 RESOURCE_LOCKED, and the
    queue vanishes when this process's connection does, so the lock can never
    outlive a crash and need manual clearing.

    Must be taken BEFORE delete_old_queues, not as a later sanity check. That
    call deletes intermediate_queue / slot_queue / rpc_queue and Server.__init__
    then re-fills the permit pool and truncates the seven result files — so a
    second server starting mid-run destroys the live run's queues, doubles the
    permit pool (breaking the RAM bound) and wipes the results before anything
    downstream could notice. It also splits fps_queue between two competing
    consumers, which halves every FPS number both of them report.

    Returns the connection; the caller MUST keep it referenced for the whole
    run, because closing it (or letting GC close it) releases the lock.
    """
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=address, port=5672, virtual_host=f"{virtual_host}",
        credentials=credentials,
        # No heartbeats: this connection carries no traffic and nobody services
        # its ioloop, so heartbeat policing would close it — releasing the lock
        # — in the middle of a perfectly healthy run.
        heartbeat=0,
    ))
    channel = connection.channel()
    try:
        channel.queue_declare(queue=SERVER_LOCK_QUEUE, durable=False, exclusive=True)
    except pika.exceptions.ChannelClosedByBroker as e:
        try:
            connection.close()
        except Exception:
            pass
        if e.reply_code == 405:   # RESOURCE_LOCKED — held by another connection
            raise RuntimeError(
                f"another server already holds '{SERVER_LOCK_QUEUE}' on "
                f"{address} vhost '{virtual_host}'. Only one server may run per "
                f"broker: stop the other server.py first. Two servers delete each "
                f"other's queues and split fps_queue, which halves every reported "
                f"FPS number."
            ) from e
        raise
    return connection


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues/{quote(virtual_host, safe="")}'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            # An exclusive queue belongs to the connection that declared it —
            # touching it from here is a 405 that closes our channel and aborts
            # the rest of the cleanup. server_lock is exactly such a queue, and
            # it is ours: we are only running because we hold it.
            if queue.get('exclusive') or queue_name == SERVER_LOCK_QUEUE:
                continue
            # The permit pools are DELETED, not purged: their x-max-length comes
            # from `a` (transport_plan), so it must be free to change between runs
            # — re-declaring a queue with different arguments is a
            # PRECONDITION_FAILED that closes the channel.
            try:
                if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                        "result") or queue_name.startswith("rpc_queue") or queue_name.startswith("bbox_queue") or queue_name.startswith("mfq") or queue_name.startswith("slot_queue"):

                    http_channel.queue_delete(queue=queue_name)

                else:
                    http_channel.queue_purge(queue=queue_name)
            except pika.exceptions.ChannelClosedByBroker:
                # One unusable queue must not abort the cleanup of the others,
                # and a closed channel cannot be reused for the next iteration.
                http_channel = connection.channel()

        connection.close()
        return True
    else:
        return False

def compute_iou(box1, box2):
    """Compute IoU"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0

def compute_ap(tp, fp, total_gt):
    tp = np.array(tp)
    fp = np.array(fp)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (total_gt + 1e-6)
    ap = 0.0
    for i in range(len(precision)):
        if i == 0 or recall[i] != recall[i - 1]:
            delta_r = recall[i] - recall[i - 1] if i > 0 else recall[i]
            ap += precision[i] * delta_r
    return ap

def compute_map(preds, gts, iou_threshold=0.1):
    from collections import defaultdict
    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(lambda: defaultdict(list))

    for img_id, cls, x1, y1, x2, y2 in gts:
        gts_by_class[int(cls)][img_id].append([x1, y1, x2, y2])

    for img_id, cls, x1, y1, x2, y2, conf in preds:
        preds_by_class[int(cls)].append((img_id, [x1, y1, x2, y2], float(conf)))

    ap_list = []
    for cls in sorted(preds_by_class.keys()):
        detections = sorted(preds_by_class[cls], key=lambda x: -x[2])
        gt_class = gts_by_class[cls]
        tp, fp = [], []
        matched = defaultdict(set)
        total_gt = sum(len(boxes) for boxes in gt_class.values())
        for img_id, box_pred, _ in detections:
            matched_gt_boxes = gt_class.get(img_id, [])
            ious = [compute_iou(box_pred, gt_box) for gt_box in matched_gt_boxes]
            if ious:
                max_iou = max(ious)
                max_idx = np.argmax(ious)
                if max_iou >= iou_threshold and max_idx not in matched[img_id]:
                    tp.append(1)
                    fp.append(0)
                    matched[img_id].add(max_idx)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)
        ap = compute_ap(tp, fp, total_gt)
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0

def load_ground_truth(label_dir, image_dir):
    gts = []
    for file in sorted(os.listdir(label_dir)):
        if not file.endswith(".txt"):
            continue
        image_id = os.path.splitext(file)[0]
        label_path = os.path.join(label_dir, file)
        img_path = os.path.join(image_dir, image_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gts.append([image_id, int(cls), x1, y1, x2, y2])
    return gts