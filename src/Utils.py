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
# of pickle framing.
FRAME_BYTES = 640 * 640 * 3 * 4          # 4,915,200 B = 4.6875 MiB

# A resident message costs the broker more than its body: the frame assembler
# holds part of the payload a second time while the body is still arriving, and
# there is per-message and queue-index overhead on top. The permit pool is
# sized against an inflated body so those transients cannot push the host past
# the budget.
BODY_OVERHEAD_FACTOR = 1.2

# Defaults for rabbit.broker-ram-budget-mb / broker-ram-reserve-mb. The reserve
# is what RabbitMQ costs before it holds any of our payload (Erlang VM +
# management plugin); measure it on your broker with the queues empty and set
# the key if it differs.
DEFAULT_RAM_BUDGET_MB = 1024
DEFAULT_RAM_RESERVE_MB = 300

# Broker-side mutex proving one server owns the run — see acquire_server_lock.
SERVER_LOCK_QUEUE = "server_lock"


def _load_config(config):
    """Return the given config dict, or read config.yaml when called with None
    (the edge/cloud Scheduler doesn't carry the config object around)."""
    if config is None:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config


def raw_frame_transport_plan(config=None):
    """How many raw-frame batches may be resident in the broker at once, derived
    from a RAM budget instead of a hand-picked message count.

    `rabbit.broker-ram-budget-mb` is the ceiling for the whole broker host, and
    `broker-ram-reserve-mb` is what the RabbitMQ runtime itself occupies, so
    only the difference is available for message bodies. The permit pool is then
    sized to keep

        permits x body_bytes x BODY_OVERHEAD_FACTOR  <=  available

    which is an invariant, not an estimate: an edge takes a permit BEFORE it
    starts transmitting, so at most `permits` bodies can ever be in flight (see
    get_publish_slots for why x-max-length alone cannot bound this).

    Pure calculation — it never raises. `permits` is 0 when the budget is
    switched off (budget 0 and no manual cap, i.e. unbounded broker memory) and
    -1 when a single batch cannot fit the budget at all; the caller decides
    whether that is fatal (validate_transport_config).

    Every machine must compute the SAME number, because it becomes
    intermediate_queue's x-max-length and a mismatched queue_declare is a
    PRECONDITION_FAILED. It is derived from server.batch-size and the rabbit
    budget keys, so those three must be identical in every config.yaml.
    """
    config = _load_config(config)
    rabbit = config.get('rabbit', {}) or {}
    batch_size = int((config.get('server') or {}).get('batch-size') or 0)
    budget_mb = float(rabbit.get('broker-ram-budget-mb', DEFAULT_RAM_BUDGET_MB) or 0)
    reserve_mb = float(rabbit.get('broker-ram-reserve-mb', DEFAULT_RAM_RESERVE_MB) or 0)
    available = max(0.0, budget_mb - reserve_mb) * _MB
    body = batch_size * FRAME_BYTES
    cost = body * BODY_OVERHEAD_FACTOR
    # max-queue-messages is kept as a manual override, and it can only ever
    # LOWER the derived count — otherwise it would be a way to opt out of the
    # RAM budget by editing an unrelated-looking key.
    override = int(rabbit.get('max-queue-messages') or 0)

    if budget_mb <= 0 or batch_size <= 0:
        permits = override
    elif cost > available:
        permits = -1
    else:
        permits = int(available // cost)
        if override:
            permits = min(permits, override)

    return {
        "batch_size": batch_size,
        "body_bytes": body,
        "body_cost_bytes": cost,
        "available_bytes": available,
        "budget_mb": budget_mb,
        "reserve_mb": reserve_mb,
        "permits": permits,
        "peak_bytes": max(permits, 0) * cost,
        "max_batch_size": int(available // (FRAME_BYTES * BODY_OVERHEAD_FACTOR)),
        "override": override,
    }


def transport_config_error(plan):
    """The reason `plan` is unusable, or None when it is fine."""
    if plan["permits"] != -1:
        return None
    return (
        f"server.batch-size {plan['batch_size']} makes one raw-frame message "
        f"{plan['body_bytes'] / _MB:.0f} MiB (~{plan['body_cost_bytes'] / _MB:.0f} MiB "
        f"of broker RAM), but only {plan['available_bytes'] / _MB:.0f} MiB is "
        f"available: {plan['budget_mb']:.0f} MB rabbit.broker-ram-budget-mb minus "
        f"{plan['reserve_mb']:.0f} MB broker-ram-reserve-mb. The broker cannot hold "
        f"even one batch inside the budget, so no permit count can keep it under. "
        f"Lower server.batch-size to {plan['max_batch_size']} or less, or raise "
        f"rabbit.broker-ram-budget-mb."
    )


def validate_transport_config(config=None):
    """Return the plan, raising RuntimeError if it cannot honour the budget.
    Call this at startup, before any queue is declared, so an impossible config
    fails with one readable line instead of a traceback out of queue_declare."""
    plan = raw_frame_transport_plan(config)
    error = transport_config_error(plan)
    if error:
        raise RuntimeError(error)
    return plan


def describe_transport_plan(plan):
    """One-line startup banner spelling out the RAM arithmetic, so a run's log
    records the bound it was actually operating under."""
    if plan["permits"] <= 0:
        return ("[Slots] disabled — rabbit.broker-ram-budget-mb is 0 and no "
                "max-queue-messages is set; broker memory is UNBOUNDED")
    peak_mb = plan["peak_bytes"] / _MB
    return (
        f"[Slots] {plan['permits']} raw-frame publish permits x "
        f"{plan['body_bytes'] / _MB:.0f} MiB/batch (batch-size {plan['batch_size']}) "
        f"-> peak {peak_mb:.0f} MiB of message bodies + {plan['reserve_mb']:.0f} MiB "
        f"broker reserve = {peak_mb + plan['reserve_mb']:.0f} MiB of "
        f"{plan['budget_mb']:.0f} MiB budget"
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
    """Overflow args for intermediate_queue / intermediate_queue_k — the heavy
    queue carrying raw image batches (~MB each). Capped at the same N as the
    permit pool (raw_frame_transport_plan), so a held permit always corresponds
    to a free slot and a permitted publish can never be NACKed."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    return _overflow_args(get_publish_slots(config),
                          rabbit.get('overflow', 'reject-publish'))


def get_bbox_queue_args(config=None):
    """Overflow args for bbox_queue — the light queue carrying edge-computed
    bboxes (~KB each, text). Capped by `bbox-max-queue-messages`, which can be
    much deeper than the image queue for the same RAM. Falls back to the
    raw-frame permit count if the bbox-specific key is unset."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    max_len = rabbit.get('bbox-max-queue-messages') or get_publish_slots(config)
    return _overflow_args(max_len, rabbit.get('overflow', 'reject-publish'))


def get_slot_queue_args(config=None):
    """Args for slot_queue — the raw-frame publish permit pool.

    The cap is the whole point: it makes the pool physically incapable of
    holding more permits than the RAM budget allows, so broker RAM stays bounded
    even when permit accounting goes wrong. It can go wrong easily — the cloud
    mints a permit for every raw-frame batch it pulls, so ANY message that was
    not paid for (a leftover from a previous run, a batch from an edge still
    running older code during a rolling restart) inflates the pool. An inflated
    pool hands out permits instantly, every edge routes to the cloud, and all 9
    push ~150 MB at once. Without this cap that failure is unbounded.

    'drop-head' rather than 'reject-publish': permits are fungible, so when the
    pool is full the right thing is to quietly discard the surplus one.
    """
    max_len = get_publish_slots(config)
    if not max_len:
        return None
    return {'x-max-length': int(max_len), 'x-overflow': 'drop-head'}


def get_publish_slots(config=None):
    """How many raw-frame batches may be in the broker at once.

    Sized by raw_frame_transport_plan from the RAM budget, and equal to the
    intermediate_queue cap deliberately: an edge takes one permit from
    slot_queue BEFORE it transmits, so holding a permit guarantees a free slot
    in the queue and the publish cannot be NACKed. That matters because a NACK
    costs a full retransmission of the body (~150 MB for raw frames).

    'x-max-length' alone cannot bound broker memory here — the broker must
    receive a message in full before it can evaluate the limit and reject it,
    so N edges can each push a complete body before any one is refused. The
    permit is taken up front, so N is bounded by the permit count instead of
    by the edge count. Returns 0 only when the budget is switched off.

    Raises rather than falling back to 0 when the budget is unsatisfiable: 0
    means "unbounded", which is the opposite of what an over-budget config asks
    for, and it would fail silently in exactly the case the budget exists for.
    """
    return max(0, validate_transport_config(config)["permits"])


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
            # slot_queue is DELETED, not purged: its x-max-length is derived from
            # the RAM budget and batch-size (raw_frame_transport_plan), so it must
            # be free to change between runs — re-declaring a queue with different
            # arguments is a PRECONDITION_FAILED that closes the channel.
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