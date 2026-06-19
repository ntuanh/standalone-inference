import pika
from requests.auth import HTTPBasicAuth
from urllib.parse import quote
import requests
import os
import numpy as np
import cv2


def _load_config(config):
    """Return the given config dict, or read config.yaml when called with None
    (the edge/cloud Scheduler doesn't carry the config object around)."""
    if config is None:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config


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
    queue carrying raw image batches (~MB each). Capped by `max-queue-messages`."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    return _overflow_args(rabbit.get('max-queue-messages'),
                          rabbit.get('overflow', 'reject-publish'))


def get_bbox_queue_args(config=None):
    """Overflow args for bbox_queue — the light queue carrying edge-computed
    bboxes (~KB each, text). Capped by `bbox-max-queue-messages`, which can be
    much deeper than the image queue for the same RAM. Falls back to
    `max-queue-messages` if the bbox-specific key is unset."""
    rabbit = _load_config(config).get('rabbit', {}) or {}
    max_len = rabbit.get('bbox-max-queue-messages', rabbit.get('max-queue-messages'))
    return _overflow_args(max_len, rabbit.get('overflow', 'reject-publish'))


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
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "result") or queue_name.startswith("rpc_queue") or queue_name.startswith("bbox_queue") or queue_name.startswith("mfq"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

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