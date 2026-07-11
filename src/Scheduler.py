import torch
import cv2
import pickle
import traceback
from tqdm import tqdm
import time
import csv
import os
import psutil
import numpy as np
import pika
import pika.exceptions

from src.Compress import Encoder,Decoder
import src.Log as Log
from src.Model import inference, postprocess_yolo
from src.Utils import get_intermediate_queue_args, get_bbox_queue_args

# Fixed cap on intermediate_queue depth (messages) before an edge waits.
# Only only_cloud sends large raw frames (~150MB/msg), which can blow up
# RabbitMQ broker memory on the Hub if too many pile up. split/Hungarian
# sends small compressed feature maps and has never overflowed, so it's
# left unconstrained.
MAX_QUEUE_ONLY_CLOUD = 15

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device

        cid_short = str(client_id).replace('-', '')[:12]
        self._timing_log_edge  = f"timing_edge_{cid_short}.log"
        self._timing_log_cloud = f"timing_cloud_{cid_short}.log"
        for tlog in [self._timing_log_edge, self._timing_log_cloud]:
            if os.path.exists(tlog):
                try:
                    os.remove(tlog)
                except Exception:
                    pass

        self.size_message = None
        self.intermediate_queue = f"intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False,
                                   arguments=get_intermediate_queue_args())
        # adaptive mode only: edge-computed bboxes go here, kept off
        # intermediate_queue so they don't skew the routing depth check.
        self.bbox_queue = "bbox_queue"
        self.channel.queue_declare(self.bbox_queue, durable=False,
                                   arguments=get_bbox_queue_args())
        # FPS ping channel: the cloud publishes one tiny "done" message here per
        # finished batch; the SERVER consumes them and computes FPS centrally
        # (see Server.on_fps_done). Unbounded — pings are a few hundred bytes.
        self.fps_queue = "fps_queue"
        self.channel.queue_declare(self.fps_queue, durable=False)
        # Whole-run utilization reports (one per device, sent after "end") go
        # here; the server collects them at shutdown into utilization.log.
        self.utilization_queue = "utilization_queue"
        self.channel.queue_declare(self.utilization_queue, durable=False)
        self._my_metrics_queue = None  # set by _setup_metrics_fanout_queue
        # Publisher confirms (enabled lazily on the edge in first_layer) let the
        # broker NACK a publish that hit the reject-publish overflow ceiling, so
        # _publish_intermediate can wait+retry instead of silently dropping it.
        self._confirms_enabled = False

        self.map_metric = None
        self.gt_dict = {}
        self._det_results = {}
        self._map_updated = False
        self._load_gt_dict()

    def get_ram_mb(self):
        try:
            import subprocess, re
            result = subprocess.run(
                ['tegrastats', '--once'],
                capture_output=True, text=True, timeout=2
            )
            m = re.search(r'RAM (\d+)/\d+MB', result.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _get_queue_depth(self, queue_name):
        """Number of ready messages in queue_name (passive declare = read-only).
        Used by adaptive routing: depth 0 on intermediate_queue means the cloud
        is keeping up, so the next batch can be offloaded for full cloud YOLO."""
        try:
            return self.channel.queue_declare(queue_name, passive=True).method.message_count
        except Exception:
            return 0

    # Connection errors that mean "the TCP link to the broker died, rebuild it".
    # A single-threaded BlockingConnection can't service I/O during multi-second
    # blocking inference, so the broker / a NAT may drop the idle socket; we
    # recover by reconnecting rather than trying to prevent the drop.
    _CONN_ERRORS = (
        pika.exceptions.StreamLostError,
        pika.exceptions.AMQPConnectionError,
        pika.exceptions.ConnectionClosed,
        pika.exceptions.ChannelWrongStateError,
        pika.exceptions.ChannelClosed,
    )

    def _rabbit_params(self):
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        r = cfg['rabbit']
        return pika.ConnectionParameters(
            host=r['address'], port=5672,
            virtual_host=f"{r['virtual-host']}",
            credentials=pika.PlainCredentials(r['username'], r['password']),
            heartbeat=0, blocked_connection_timeout=600,
        )

    def _reconnect(self):
        """Tear down the dead connection and build a fresh one + channel, then
        restore the queue declarations and publisher confirms so the caller can
        retry. Retries until the broker is reachable again."""
        Log.print_with_color("[Reconnect] Connection lost — rebuilding RabbitMQ link...", "yellow")
        # Best-effort close of the old, broken objects.
        try:
            old_conn = self.channel.connection
        except Exception:
            old_conn = None
        for obj in (self.channel, old_conn):
            try:
                if obj is not None and obj.is_open:
                    obj.close()
            except Exception:
                pass

        while True:
            try:
                connection = pika.BlockingConnection(self._rabbit_params())
                self.channel = connection.channel()
                # Re-declare the queues this client touches (idempotent; args must
                # match the original declares or the broker rejects them).
                self.channel.queue_declare(self.intermediate_queue, durable=False,
                                           arguments=get_intermediate_queue_args())
                self.channel.queue_declare(self.bbox_queue, durable=False,
                                           arguments=get_bbox_queue_args())
                self.channel.queue_declare(self.fps_queue, durable=False)
                self.channel.queue_declare(self.utilization_queue, durable=False)
                if self._confirms_enabled:
                    self._confirms_enabled = False
                    self._enable_publisher_confirms()
                Log.print_with_color("[Reconnect] RabbitMQ link re-established.", "green")
                return
            except Exception as e:
                Log.print_with_color(f"[Reconnect] Retry in 1s ({e})", "yellow")
                time.sleep(1.0)

    def _check_backpressure(self):
        max_queue = MAX_QUEUE_ONLY_CLOUD
        depth = self.channel.queue_declare(self.intermediate_queue, passive=True).method.message_count
        if depth < max_queue:
            return

        Log.print_with_color(
            f"[BackPressure] '{self.intermediate_queue}' depth={depth} >= max_queue={max_queue}, waiting", "yellow")
        while depth >= max_queue:
            time.sleep(0.1)
            depth = self.channel.queue_declare(self.intermediate_queue, passive=True).method.message_count
        Log.print_with_color(
            f"[BackPressure] '{self.intermediate_queue}' depth={depth} < max_queue={max_queue}, resuming", "green")

    def _enable_publisher_confirms(self):
        """Turn the edge publish channel into confirm mode so the broker tells us
        (via basic.nack) when a publish is rejected by the reject-publish overflow
        policy. Idempotent — safe to call once per run."""
        if self._confirms_enabled:
            return
        try:
            self.channel.confirm_delivery()
            self._confirms_enabled = True
        except Exception as e:
            # Confirms already on, or broker doesn't support them — fall back to
            # the depth-poll back-pressure, which still bounds the queue.
            Log.print_with_color(f"[Overflow] confirm_delivery() not enabled: {e}", "yellow")

    def _publish_intermediate(self, queue_name, body):
        """Publish one batch to intermediate_queue with broker overflow handling.

        With publisher confirms on and the queue declared 'x-overflow:
        reject-publish', a publish that would exceed 'x-max-length' is NACKed by
        the broker (pika raises NackError) and the message is NOT enqueued. We
        treat that as back-pressure: wait for the cloud to drain, then retry —
        so the frame is never lost. Mirrors the depth-poll _check_backpressure,
        but uses the broker as the source of truth and works in every mode
        (split / only_edge / only_cloud), not just only_cloud."""
        while True:
            try:
                self.channel.basic_publish(exchange='', routing_key=queue_name, body=body)
                return
            except (pika.exceptions.NackError, pika.exceptions.UnroutableError):
                Log.print_with_color(
                    f"[Overflow] '{queue_name}' full — broker rejected publish "
                    f"(reject-publish). Waiting for cloud to drain...", "yellow")
                # Use connection.sleep (not time.sleep) so heartbeats / socket I-O
                # keep flowing while we block — otherwise pika's BlockingConnection
                # goes silent and the broker resets it (StreamLostError).
                try:
                    self.channel.connection.sleep(0.1)
                except Exception:
                    time.sleep(0.1)
            except self._CONN_ERRORS:
                # Broker reset the idle socket (e.g. during long inference) —
                # rebuild the connection and retry the same body (no loss).
                self._reconnect()

    def write_metrics(self, mode, role, best_cut, batch_id, batch_size, latency_ms, fps, ram_mb, message_size_bytes=0, e2e_latency_ms=0, edge_start_time=None, inference_path=""):
        file_path = f"metrics_raw_{self.intermediate_queue}_{str(self.client_id).replace('-', '')}.csv"
        file_exists = os.path.exists(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "mode",
                    "role",
                    "best_cut",
                    "batch_id",
                    "batch_size",
                    "latency_ms",
                    "fps",
                    "ram_mb",
                    "message_size_bytes",
                    "e2e_latency_ms",
                    "edge_start_time",
                    "inference_path",
                ])

            writer.writerow([
                mode,
                role,
                best_cut,
                batch_id,
                batch_size,
                round(latency_ms, 3),
                round(fps, 3) if fps > 0 else "",  # fps=0 (first batch) → empty
                round(ram_mb, 3),
                message_size_bytes,
                round(e2e_latency_ms, 3),
                edge_start_time if edge_start_time is not None else "",
                inference_path,
            ])

    def _setup_metrics_fanout_queue(self):
        """Cloud client gọi trước khi inference: tạo queue riêng bind vào fanout exchange.
        Mỗi cloud nhận một bản copy metrics từ tất cả edge trong cluster."""
        exchange = f"metrics_fanout_{self.intermediate_queue}"
        my_queue = f"mfq_{str(self.client_id).replace('-', '')}"
        try:
            self.channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=False)
            self.channel.queue_declare(my_queue, durable=False)
            self.channel.queue_bind(queue=my_queue, exchange=exchange)
            self._my_metrics_queue = my_queue
        except Exception as e:
            Log.print_with_color(f"[Metrics] Fanout setup failed: {e}", "yellow")
            self._my_metrics_queue = None

    def send_next_layer(self, intermediate_queue, data, compress):

        if compress["enable"]:
            data["data"] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
            data["data"], data["shape"] = Encoder(data_output=data["data"], num_bits=compress["num_bit"])

        else:
            data["data"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
        message = pickle.dumps({
            "action": "OUTPUT",
            "data": data
        })
        self.size_message = len(message)

        self._publish_intermediate(intermediate_queue, message)

    def _load_gt_dict(self, gt_dir="datasets/groundtruth"):
        if not os.path.isdir(gt_dir):
            return
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self.map_metric = MeanAveragePrecision(iou_type="bbox")
            self.map_metric.warn_on_many_detections = False
        except ImportError:
            Log.print_with_color("[!] torchmetrics not installed, mAP disabled", "red")
            return
        for fname in sorted(os.listdir(gt_dir)):
            if not fname.endswith(".txt"):
                continue
            try:
                num = int(os.path.splitext(fname)[0].split("_")[-1])
            except ValueError:
                continue
            boxes, labels = [], []
            with open(os.path.join(gt_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    boxes.append([(cx - bw/2)*640, (cy - bh/2)*640,
                                  (cx + bw/2)*640, (cy + bh/2)*640])
                    labels.append(int(cls))
            self.gt_dict[num] = {
                "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
            }
        Log.print_with_color(f"[mAP] Loaded GT for {len(self.gt_dict)} frames from '{gt_dir}'", "green")

    def _update_map(self, batch_results, batch_id, batch_size, map_results=None):
        import json
        self._map_updated = True
        # map_results uses conf≈0.001 so torchmetrics gets the full PR curve;
        # batch_results (conf=0.25) is only for the detection stream / display.
        _map = map_results if map_results is not None else batch_results
        for img_idx, (r, rm) in enumerate(zip(batch_results, _map)):
            frame_num = batch_id * batch_size + img_idx + 1
            dets = [
                {
                    "box":   r["boxes"][i].cpu().tolist(),
                    "score": round(float(r["scores"][i]), 4),
                    "class": int(r["classes"][i]),
                }
                for i in range(len(r["boxes"]))
            ]
            self._det_results[frame_num] = dets
            with open("detections_stream.jsonl", "a") as f:
                f.write(json.dumps({"frame": frame_num, "dets": dets}) + "\n")
            if self.map_metric is None or frame_num not in self.gt_dict:
                continue
            self.map_metric.update(
                [{"boxes":  rm["boxes"].cpu().float(),
                  "scores": rm["scores"].cpu().float(),
                  "labels": rm["classes"].cpu().long()}],
                [self.gt_dict[frame_num]]
            )

    def _print_map(self):
        if self.map_metric is None:
            Log.print_with_color("[mAP] Skipped: groundtruth not found on this device (datasets/groundtruth/ missing)", "yellow")
            return
        if not self.gt_dict:
            Log.print_with_color("[mAP] Skipped: groundtruth folder exists but no valid .txt files loaded", "yellow")
            return
        try:
            result = self.map_metric.compute()
            print("=" * 50)
            print(f"  [mAP]   mAP@50={result['map_50']:.4f}  mAP@50:95={result['map']:.4f}")
            print("=" * 50)
        except Exception as e:
            Log.print_with_color(f"[mAP] compute failed: {e}", "red")

    def _write_detections_json(self):
        import json
        out = "detections.json"
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in sorted(self._det_results.items())}, f)
        Log.print_with_color(f"[Tracker] Saved {out} ({len(self._det_results)} frames)", "green")

    def _compute_utilization(self, log_path, role):
        """Read a timing log back ("<ns> start / get input / output / end") and
        compute ONE whole-run utilization for this device: total busy time
        (sum over every 'get input' -> 'output' interval) / total time
        ('start' -> 'end'). Extra events like queue_wait_start/end are ignored.
        Returns a stats dict for _send_utilization, or None on a bad log."""
        if not os.path.exists(log_path):
            return None
        t_start = t_end = t_input = None
        busy_ns = 0
        n_packages = 0
        try:
            with open(log_path) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2 or not parts[0].isdigit():
                        continue
                    ts, event = int(parts[0]), parts[1]
                    if event == "start":
                        t_start = ts
                    elif event == "get input":
                        t_input = ts
                    elif event == "output":
                        if t_input is not None:
                            busy_ns += ts - t_input
                            n_packages += 1
                            t_input = None
                    elif event == "end":
                        t_end = ts
        except Exception as e:
            Log.print_with_color(f"[Utilization][{role}] parse failed for {log_path}: {e}", "yellow")
            return None
        if t_start is None or t_end is None or t_end <= t_start:
            Log.print_with_color(f"[Utilization][{role}] incomplete log {log_path}, skipped", "yellow")
            return None
        total_ns = t_end - t_start
        util = busy_ns / total_ns
        Log.print_with_color(
            f"[Utilization][{role}] packages={n_packages} "
            f"busy={busy_ns / 1e9:.3f}s total={total_ns / 1e9:.3f}s "
            f"utilization={util * 100:.2f}%", "green")
        return {
            "role": role,
            "packages": n_packages,
            "busy_ns": busy_ns,
            "total_ns": total_ns,
            "utilization": util,
        }

    def _send_utilization(self, stats):
        """Publish this device's whole-run utilization report to the server
        (utilization_queue); the server appends it to its utilization log."""
        if stats is None:
            return
        body = pickle.dumps({
            "action": "UTILIZATION",
            "client_id": self.client_id,
            "layer_id": self.layer_id,
            **stats,
        })
        for attempt in range(2):
            try:
                self.channel.queue_declare(self.utilization_queue, durable=False)
                self.channel.basic_publish(
                    exchange='', routing_key=self.utilization_queue, body=body)
                return
            except self._CONN_ERRORS:
                if attempt == 0:
                    self._reconnect()
            except Exception as e:
                Log.print_with_color(f"[Utilization] send failed: {e}", "yellow")
                return
        Log.print_with_color("[Utilization] send failed after reconnect", "yellow")

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def first_layer(self, model, data, batch_size, splits, logger, compress, mode="split", save_set=None):
        input_image = []
        # Edge is the only publisher to intermediate_queue → enable confirms so a
        # reject-publish overflow NACK surfaces in _publish_intermediate.
        self._enable_publisher_confirms()
        if mode != "only_cloud":
            model.eval()
            model.to(self.device)

        video_path = data
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            Log.print_with_color(f"Not open video", "red")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        batch_id = 0
        prev_batch_end = None
        with open(self._timing_log_edge, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 640))
            frame = frame.astype('float32') / 255.0
            tensor = torch.from_numpy(frame).permute(2, 0, 1)  # shape: (3, 640, 640)
            input_image.append(tensor)

            if len(input_image) == batch_size:
                t_batch_ready = time.perf_counter()
                gap_ms = (t_batch_ready - prev_batch_end) * 1000 if prev_batch_end is not None else 0.0
                with open(self._timing_log_edge, "a") as _tf:
                    print(str(time.time_ns()) + " get input", file=_tf)
                batch_start = time.perf_counter()
                edge_start_wall = time.time()

                _stack_start = time.perf_counter()
                input_image = torch.stack(input_image)
                if mode not in ("only_cloud", "adaptive"):
                    # only_cloud / adaptive: edge may skip GPU inference (cloud path),
                    # so keep frames on CPU and move to device only if we route locally,
                    # avoiding a wasted CPU->GPU->CPU round trip before sending.
                    input_image = input_image.to(self.device)
                stack_ms = (time.perf_counter() - _stack_start) * 1000

                inference_ms = 0.0
                queue_wait_ms = 0.0
                send_ms = 0.0
                route_path = None   # adaptive: "split" (cloud) or "edge_only"

                # ===== ONLY CLOUD =====
                if mode == "only_cloud":
                    frames_cpu = input_image
                    y = {
                        "data": [frames_cpu[i].clone() for i in range(len(frames_cpu))],
                        "width": width,
                        "height": height,
                        "edge_start_time": edge_start_wall
                    }

                    _wait_start = time.perf_counter()
                    with open(self._timing_log_edge, "a") as _tf:
                        print(str(time.time_ns()) + " queue_wait_start", file=_tf)
                    self._check_backpressure()
                    with open(self._timing_log_edge, "a") as _tf:
                        print(str(time.time_ns()) + " queue_wait_end", file=_tf)
                    queue_wait_ms = (time.perf_counter() - _wait_start) * 1000

                    _send_start = time.perf_counter()
                    self.send_next_layer(
                        self.intermediate_queue,
                        y,
                        {"enable": False}
                    )
                    send_ms = (time.perf_counter() - _send_start) * 1000

                # ===== ONLY EDGE =====
                elif mode == "only_edge":

                    _inf_start = time.perf_counter()
                    y = []
                    with torch.no_grad():
                        x, y = inference(model, input_image, y, 0, save_set)
                    inference_ms = (time.perf_counter() - _inf_start) * 1000

                    results     = postprocess_yolo(x, conf_thres=0.25,  iou_thres=0.5)
                    map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=0.5)
                    self._update_map(results, batch_id, batch_size, map_results=map_results)

                    _send_start = time.perf_counter()
                    payload = {
                        "width": width,
                        "height": height,
                        "results": [
                            {
                                "boxes":   r["boxes"].cpu().numpy(),
                                "scores":  r["scores"].cpu().numpy(),
                                "classes": r["classes"].cpu().numpy(),
                            }
                            for r in results
                        ],
                        "edge_start_time": edge_start_wall,
                    }
                    body = pickle.dumps({"action": "OUTPUT", "data": payload})
                    self.size_message = len(body)
                    self._publish_intermediate(self.intermediate_queue, body)
                    send_ms = (time.perf_counter() - _send_start) * 1000

                # ===== ADAPTIVE (full cloud  OR  full edge, decided per batch) =====
                elif mode == "adaptive":
                    depth = self._get_queue_depth(self.intermediate_queue)
                    if depth == 0:
                        # --- Cloud has capacity → offload raw frames (full cloud YOLO) ---
                        route_path = "split"
                        frames_cpu = input_image  # still on CPU
                        y = {
                            "data": [frames_cpu[i].clone() for i in range(len(frames_cpu))],
                            "width": width,
                            "height": height,
                            "edge_start_time": edge_start_wall,
                        }
                        _wait_start = time.perf_counter()
                        self._check_backpressure()
                        queue_wait_ms = (time.perf_counter() - _wait_start) * 1000

                        _send_start = time.perf_counter()
                        self.send_next_layer(self.intermediate_queue, y, {"enable": False})
                        send_ms = (time.perf_counter() - _send_start) * 1000
                    else:
                        # --- Cloud backlogged → run full YOLO locally, ship bboxes ---
                        route_path = "edge_only"
                        input_image = input_image.to(self.device)

                        _inf_start = time.perf_counter()
                        y = []
                        with torch.no_grad():
                            x, y = inference(model, input_image, y, 0, save_set)
                        inference_ms = (time.perf_counter() - _inf_start) * 1000

                        results     = postprocess_yolo(x, conf_thres=0.25,  iou_thres=0.5)
                        map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=0.5)
                        self._update_map(results, batch_id, batch_size, map_results=map_results)

                        _send_start = time.perf_counter()
                        payload = {
                            "width": width,
                            "height": height,
                            "results": [
                                {
                                    "boxes":   r["boxes"].cpu().numpy(),
                                    "scores":  r["scores"].cpu().numpy(),
                                    "classes": r["classes"].cpu().numpy(),
                                }
                                for r in results
                            ],
                            "edge_start_time": edge_start_wall,
                        }
                        body = pickle.dumps({"action": "OUTPUT", "data": payload})
                        self.size_message = len(body)
                        self._publish_intermediate(self.bbox_queue, body)
                        send_ms = (time.perf_counter() - _send_start) * 1000

                # ===== SPLIT INFERENCE =====
                else:

                    _inf_start = time.perf_counter()
                    y = []
                    with torch.no_grad():
                        x, y = inference(model, input_image, y, 0, save_set)
                    y[-1] = x
                    inference_ms = (time.perf_counter() - _inf_start) * 1000

                    y = {
                        "data": y,
                        "width": width,
                        "height": height,
                        "edge_start_time": edge_start_wall
                    }

                    _send_start = time.perf_counter()
                    self.send_next_layer(
                        self.intermediate_queue,y,compress
                    )
                    send_ms = (time.perf_counter() - _send_start) * 1000
                batch_end = time.perf_counter()
                with open(self._timing_log_edge, "a") as _tf:
                    print(str(time.time_ns()) + " output", file=_tf)
                latency_ms = (batch_end - batch_start) * 1000
                # FPS = frames / time spent actually processing this batch (not the
                # gap since the last batch), so idle/capture waits don't distort it.
                _proc_s = batch_end - batch_start
                fps = batch_size / _proc_s if _proc_s > 0 else 0.0
                e2e_latency_ms = 0.0
                _ram_start = time.perf_counter()
                ram_mb = self.get_ram_mb()
                ram_ms = (time.perf_counter() - _ram_start) * 1000
                msg_size = self.size_message if self.size_message is not None else 0

                if mode == "adaptive":
                    # split route → edge only sent frames; edge_only → edge ran YOLO.
                    edge_role = "edge_sender" if route_path == "split" else "edge"
                elif mode == "only_cloud":
                    edge_role = "edge_sender"
                else:
                    edge_role = "edge"

                _write_start = time.perf_counter()
                self.write_metrics(
                    mode=mode,
                    role=edge_role,
                    best_cut="N/A" if splits is None else splits,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    fps=fps,
                    ram_mb=ram_mb,
                    message_size_bytes=msg_size,
                    e2e_latency_ms=e2e_latency_ms,
                    edge_start_time=edge_start_wall,
                    inference_path=route_path or "",
                )
                write_ms = (time.perf_counter() - _write_start) * 1000

                # Bare "DONE" ping → fps_queue when the EDGE is the tier that
                # completed this batch (only_edge, or adaptive routed edge_only).
                # In split/only_cloud the cloud finishes the batch and pings
                # instead — exactly one ping per batch system-wide. The server
                # computes system FPS as batch_size / delta between DONEs.
                if mode == "only_edge" or route_path == "edge_only":
                    try:
                        self.channel.basic_publish(
                            exchange='', routing_key=self.fps_queue, body=b"DONE")
                    except Exception as e:
                        Log.print_with_color(f"[FPS] ping publish failed: {e}", "yellow")

                batch_interval_ms = (batch_end - prev_batch_end) * 1000 if prev_batch_end is not None else 0.0
                Log.print_with_color(
                    f"[Timing][edge] gap={gap_ms:.1f}ms stack={stack_ms:.1f}ms "
                    f"inference={inference_ms:.1f}ms queue_wait={queue_wait_ms:.1f}ms send={send_ms:.1f}ms "
                    f"ram={ram_ms:.1f}ms write={write_ms:.1f}ms "
                    f"| latency={latency_ms:.1f}ms | batch_interval={batch_interval_ms:.1f}ms",
                    "magenta"
                )

                batch_id += 1
                prev_batch_end = batch_end

                input_image = []
                pbar.update(batch_size)
            else:
                continue
        with open(self._timing_log_edge, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)
        self._send_utilization(self._compute_utilization(self._timing_log_edge, "edge"))
        print(f'size message: {self.size_message} bytes.')
        cap.release()
        pbar.close()

        # Broadcast metrics CSV lên tất cả cloud trong cluster qua fanout exchange
        metrics_file = f"metrics_raw_{self.intermediate_queue}_{str(self.client_id).replace('-', '')}.csv"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'rb') as f:
                    metrics_data = f.read()
                exchange = f"metrics_fanout_{self.intermediate_queue}"
                self.channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=False)
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key='',
                    body=pickle.dumps({"action": "METRICS", "filename": os.path.basename(metrics_file), "data": metrics_data})
                )
                Log.print_with_color(f"[Metrics] Broadcast metrics via fanout ({len(metrics_data)} bytes)", "cyan")
            except Exception as e:
                Log.print_with_color(f"[Metrics] Failed to send metrics: {e}", "yellow")

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!"}

        self.send_to_server(notify_data)

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:

                received_data = pickle.loads(body)
                Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "STOP":
                    Log.print_with_color("[>>>] Finish!", "red")
                    break
            time.sleep(0.5)


    def last_layer(self, model, batch_size, splits, logger, compress, mode="split", save_set=None):
        if mode != "only_edge":
            model.eval()
            model.to(self.device)

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        batch_id = 0
        prev_batch_end = None
        with open(self._timing_log_cloud, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)
        while True:
            try:
                method_frame, header_frame, body = self.channel.basic_get(queue=self.intermediate_queue, auto_ack=True)
                src_queue = "intermediate"
                # adaptive: if no raw batch to run, drain edge-computed bboxes (metrics only).
                # intermediate_queue is checked first so cloud YOLO always has priority.
                if not (method_frame and body) and mode == "adaptive":
                    method_frame, header_frame, body = self.channel.basic_get(queue=self.bbox_queue, auto_ack=True)
                    src_queue = "bbox"
            except self._CONN_ERRORS:
                # Connection dropped (e.g. during the previous long inference) —
                # reconnect and retry the get on the next loop iteration.
                self._reconnect()
                continue
            if method_frame and body:
                t_batch_ready = time.perf_counter()
                gap_ms = (t_batch_ready - prev_batch_end) * 1000 if prev_batch_end is not None else 0.0
                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " get input", file=_tf)
                batch_start = time.perf_counter()
                received_message_size = len(body)
                received_data = pickle.loads(body)
                y = received_data["data"]
                edge_start_time = y.get("edge_start_time", time.time())

                # adaptive routes per message: a raw batch on intermediate_queue runs
                # full cloud YOLO ('split'); a bbox message on bbox_queue is metrics
                # only ('edge_only'). eff maps it onto the existing only_cloud/only_edge
                # code paths; cloud_path is the inference_path label for metrics.
                if mode == "adaptive":
                    eff = "only_cloud" if src_queue == "intermediate" else "only_edge"
                    cloud_path = "split" if src_queue == "intermediate" else "edge_only"
                else:
                    eff = mode
                    cloud_path = ""

                # ===== ONLY EDGE (cloud just receives lightweight results) =====
                if eff == "only_edge":
                    decode_ms = 0.0
                    inference_ms = 0.0
                # ===== ONLY CLOUD =====
                elif eff == "only_cloud":
                    _decode_start = time.perf_counter()
                    input_tensor = y["data"]

                    if isinstance(input_tensor, list):
                        input_tensor = torch.stack(input_tensor)

                    input_tensor = input_tensor.to(self.device)
                    decode_ms = (time.perf_counter() - _decode_start) * 1000

                    _inf_start = time.perf_counter()
                    with torch.no_grad():
                        x, _ = inference(model, input_tensor, [], 0, save_set)
                    inference_ms = (time.perf_counter() - _inf_start) * 1000
                # ===== SPLIT INFERENCE =====
                else:
                    _decode_start = time.perf_counter()
                    if compress["enable"]:
                        y["data"] = Decoder(y["data"], y["shape"])

                        y["data"] = [
                            torch.from_numpy(t) if t is not None else None
                            for t in y["data"]
                        ]

                    y["data"] = [
                        t.to(self.device) if t is not None else None
                        for t in y["data"]
                    ]

                    list_output = y["data"]

                    x = list_output[-1]
                    decode_ms = (time.perf_counter() - _decode_start) * 1000

                    _inf_start = time.perf_counter()
                    with torch.no_grad():
                        x, _ = inference(model, x, list_output, splits, save_set)
                    inference_ms = (time.perf_counter() - _inf_start) * 1000

                if eff == "only_edge":
                    postprocess_ms = 0.0
                else:
                    _post_start = time.perf_counter()
                    results     = postprocess_yolo(x, conf_thres=0.25,  iou_thres=0.5)
                    map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=0.5)
                    self._update_map(results, batch_id, batch_size, map_results=map_results)
                    postprocess_ms = (time.perf_counter() - _post_start) * 1000

                batch_end = time.perf_counter()
                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " output", file=_tf)
                cloud_end_wall = time.time()
                latency_ms = (batch_end - batch_start) * 1000
                # FPS from processing time only. For edge_only batches the cloud did
                # no inference (just logged a bbox) → fps is meaningless, leave it 0
                # so it can't inflate the cloud average; the edge row holds the real fps.
                _proc_s = batch_end - batch_start
                if cloud_path == "edge_only":
                    fps = 0.0
                else:
                    fps = batch_size / _proc_s if _proc_s > 0 else 0.0
                e2e_latency_ms = (cloud_end_wall - edge_start_time) * 1000
                _ram_start = time.perf_counter()
                ram_mb = self.get_ram_mb()
                ram_ms = (time.perf_counter() - _ram_start) * 1000

                _write_start = time.perf_counter()
                self.write_metrics(
                    mode=mode,
                    role="cloud",
                    best_cut="N/A" if splits is None else splits,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    fps=fps,
                    ram_mb=ram_mb,
                    message_size_bytes=received_message_size,
                    e2e_latency_ms=e2e_latency_ms,
                    edge_start_time=edge_start_time,
                    inference_path=cloud_path,
                )
                write_ms = (time.perf_counter() - _write_start) * 1000

                # Bare "DONE" ping → fps_queue when the CLOUD is the tier that
                # completed this batch (split / only_cloud / adaptive split-path).
                # only_edge results and adaptive bbox messages are skipped — the
                # edge already pinged those batches, keeping the system at
                # exactly one ping per batch so the server's 1/delta FPS holds.
                if eff != "only_edge":
                    try:
                        self.channel.basic_publish(
                            exchange='', routing_key=self.fps_queue, body=b"DONE")
                    except Exception as e:
                        Log.print_with_color(f"[FPS] ping publish failed: {e}", "yellow")

                batch_interval_ms = (batch_end - prev_batch_end) * 1000 if prev_batch_end is not None else 0.0
                Log.print_with_color(
                    f"[Timing][cloud] gap={gap_ms:.1f}ms decode={decode_ms:.1f}ms "
                    f"inference={inference_ms:.1f}ms postprocess={postprocess_ms:.1f}ms "
                    f"ram={ram_ms:.1f}ms write={write_ms:.1f}ms "
                    f"| latency={latency_ms:.1f}ms | batch_interval={batch_interval_ms:.1f}ms",
                    "magenta"
                )

                batch_id += 1
                prev_batch_end = batch_end

                pbar.update(batch_size)

            else:
                broadcast_queue_name = f'reply_{self.client_id}'
                try:
                    method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                except self._CONN_ERRORS:
                    self._reconnect()
                    continue
                if body:
                    received_data = pickle.loads(body)
                    Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "STOP":
                        Log.print_with_color("[>>>] Finish!", "red")
                        break
                else:
                    time.sleep(0.5)

        with open(self._timing_log_cloud, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)
        self._send_utilization(self._compute_utilization(self._timing_log_cloud, "cloud"))
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pbar.close()

    def middle_layer(self, model):
        pass

    def _pivot_and_save(self):
        import glob as _glob

        # Namespaced theo intermediate_queue: mỗi cluster Hungarian (intermediate_queue_k)
        # pivot độc lập, không xóa/ghi đè file của cluster khác chạy chung thư mục.
        lock_path = f"metrics_pivot_{self.intermediate_queue}.lock"
        out_path = f"metrics_pivoted_{self.intermediate_queue}.csv"
        raw_glob = f"metrics_raw_{self.intermediate_queue}_*.csv"

        # Chỉ 1 client thắng lock mới làm pivot (atomic exclusive create)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return  # Client khác đang làm pivot

        # Đợi các client còn lại ghi xong hàng cuối
        time.sleep(2.0)

        # Thu thập metrics CSV từ personal fanout queue (mỗi cloud có bản copy riêng)
        my_q = self._my_metrics_queue
        if my_q:
            try:
                while True:
                    method_frame, _, body = self.channel.basic_get(queue=my_q, auto_ack=True)
                    if not method_frame:
                        break
                    msg = pickle.loads(body)
                    if msg.get("action") == "METRICS":
                        fname = msg["filename"]
                        with open(fname, 'wb') as f:
                            f.write(msg["data"])
                        Log.print_with_color(f"[Metrics] Received remote metrics: {fname}", "cyan")
            except Exception as e:
                Log.print_with_color(f"[Metrics] Warning collecting remote metrics: {e}", "yellow")

        edge_rows = []
        cloud_rows = []

        edge_seq_counter = 0
        cloud_seq_counter = 0

        for fpath in sorted(_glob.glob(raw_glob)):
            with open(fpath, newline="") as f:
                rows_in_file = list(csv.DictReader(f))
            if not rows_in_file:
                continue
            role = rows_in_file[0]["role"]
            if role in ("edge", "edge_sender"):
                edge_seq_counter += 1
                for row in rows_in_file:
                    row["device_seq"] = edge_seq_counter
                    edge_rows.append(row)
            elif role == "cloud":
                cloud_seq_counter += 1
                for row in rows_in_file:
                    row["device_seq"] = cloud_seq_counter
                    cloud_rows.append(row)

        # Join edge ↔ cloud bằng edge_start_time (timestamp edge nhúng vào mỗi message)
        edge_by_time = {
            row["edge_start_time"]: row
            for row in edge_rows
            if row.get("edge_start_time")
        }
        matched_pairs = []
        matched_edge_times = set()
        for c in cloud_rows:
            t = c.get("edge_start_time", "")
            e = edge_by_time.get(t, {})
            matched_pairs.append((e, c))
            if t:
                matched_edge_times.add(t)
        # Edge rows không có cloud tương ứng (only_edge mode)
        for e in edge_rows:
            if e.get("edge_start_time", "") not in matched_edge_times:
                matched_pairs.append((e, {}))
        # Sắp xếp theo edge_start_time tăng dần
        matched_pairs.sort(key=lambda p: float(p[0].get("edge_start_time") or p[1].get("edge_start_time") or 0))

        n_rows = len(matched_pairs)
        fieldnames = [
            "batch_id", "batch_size", "best_cut", "inference_path",
            "edge_device", "edge_latency_ms", "edge_fps", "edge_ram_mb", "edge_message_size_bytes",
            "cloud_device", "cloud_arrival_order", "cloud_latency_ms", "cloud_fps", "cloud_ram_mb", "cloud_message_size_bytes",
            "e2e_latency_ms",
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, (e, c) in enumerate(matched_pairs):
                writer.writerow({
                    "batch_id":                i,
                    "batch_size":              e.get("batch_size") or c.get("batch_size", ""),
                    "best_cut":                e.get("best_cut")   or c.get("best_cut", ""),
                    "inference_path":          e.get("inference_path") or c.get("inference_path", ""),
                    "edge_device":             e.get("device_seq", ""),
                    "edge_latency_ms":         e.get("latency_ms", ""),
                    "edge_fps":                e.get("fps", ""),
                    "edge_ram_mb":             e.get("ram_mb", ""),
                    "edge_message_size_bytes": e.get("message_size_bytes", ""),
                    "cloud_device":            c.get("device_seq", ""),
                    "cloud_arrival_order":     c.get("batch_id", ""),
                    "cloud_latency_ms":        c.get("latency_ms", ""),
                    "cloud_fps":               c.get("fps", ""),
                    "cloud_ram_mb":            c.get("ram_mb", ""),
                    "cloud_message_size_bytes":c.get("message_size_bytes", ""),
                    "e2e_latency_ms":          c.get("e2e_latency_ms") or e.get("e2e_latency_ms", ""),
                })

        for fpath in _glob.glob(raw_glob):
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

        def avg(rows, key, skip_zero_fps=False):
            filtered = rows
            if skip_zero_fps:
                filtered = [r for r in rows if float(r.get("fps") or 0) > 0]
            vals = [float(r[key]) for r in filtered if r.get(key)]
            return round(sum(vals) / len(vals), 3) if vals else None

        def wallclock_throughput(rows):
            # True system throughput = total frames handled / wall-clock span of the
            # run. Unfakeable: averaging per-batch instantaneous fps inflates wildly
            # when some batches are metrics-only (tiny processing time). Each final
            # row = one batch; cloud_end_wall = edge_start_time + e2e_latency_ms.
            starts, ends, frames = [], [], 0
            for r in rows:
                st = r.get("edge_start_time")
                bs = r.get("batch_size")
                if not st or not bs:
                    continue
                st = float(st)
                starts.append(st)
                frames += int(float(bs))
                e2e = r.get("e2e_latency_ms")
                ends.append(st + float(e2e) / 1000.0 if e2e else st)
            if not starts or not ends:
                return None
            span = max(ends) - min(starts)
            return round(frames / span, 3) if span > 0 else None

        def fps_avg_inferring(rows, exclude_path):
            # Average per-batch fps over only the rows where this tier actually ran
            # inference (drop the path the tier didn't compute). Non-adaptive modes
            # have inference_path="" so nothing is excluded — behaviour unchanged.
            sel = [r for r in rows if (r.get("inference_path") or "") != exclude_path]
            return avg(sel, "fps", True)

        def per_device_inference_fps(rows, exclude_path):
            # {device_seq: mean inference FPS} over ONLY that device's real inference
            # batches. exclude_path drops the metrics-only path for that tier:
            #   cloud → drop 'edge_only' (bbox_queue: no YOLO, metrics only)
            #   edge  → drop 'split'     (intermediate_queue: edge only sent frames)
            by_dev = {}
            for r in rows:
                if (r.get("inference_path") or "") == exclude_path:
                    continue
                seq = r.get("device_seq")
                f = float(r.get("fps") or 0)
                if seq is not None and f > 0:
                    by_dev.setdefault(seq, []).append(f)
            return {seq: round(sum(v) / len(v), 3) for seq, v in sorted(by_dev.items()) if v}

        def mb(val):
            return round(val / 1024 / 1024, 3) if val is not None else "N/A"

        cuts = set(r.get("best_cut", "N/A") for r in (edge_rows or cloud_rows))
        cut_str = "/".join(sorted(str(c) for c in cuts))
        all_rows = cloud_rows if cloud_rows else edge_rows
        final_rows = cloud_rows if cloud_rows else edge_rows
        system_fps = wallclock_throughput(final_rows)
        valid_batches = len([r for r in final_rows if r.get("edge_start_time")])
        # Edge metrics chỉ tính trên batch có cloud match (batch kia tính ở cloud kia)
        # Fallback về tất cả edge_rows nếu không có cloud (only_edge mode)
        matched_edge_rows = [e for e, c in matched_pairs if c and e]
        summary_edge_rows = matched_edge_rows if cloud_rows else edge_rows
        print("=" * 50)
        print(f"  SUMMARY  |  batches={n_rows} (valid={valid_batches})  cut={cut_str}")
        print("=" * 50)
        print(f"  [EDGE]  latency={avg(summary_edge_rows,'latency_ms',True)} ms  fps={fps_avg_inferring(summary_edge_rows,'split')}  ram={avg(summary_edge_rows,'ram_mb',True)} MB  msg={mb(avg(summary_edge_rows,'message_size_bytes'))} MB")
        print(f"  [CLOUD] latency={avg(cloud_rows,'latency_ms',True)} ms  fps={fps_avg_inferring(cloud_rows,'edge_only')}  ram={avg(cloud_rows,'ram_mb',True)} MB  msg={mb(avg(cloud_rows,'message_size_bytes'))} MB")
        print(f"  [E2E]   latency={avg(all_rows,'e2e_latency_ms',True)} ms")
        # Per-device inference FPS (intermediate_queue = cloud YOLO; bbox_queue
        # edge_only batches are metrics-only and excluded).
        edge_dev_fps  = per_device_inference_fps(edge_rows,  "split")
        cloud_dev_fps = per_device_inference_fps(cloud_rows, "edge_only")
        for seq, f in edge_dev_fps.items():
            print(f"  [EDGE  dev {seq}] inference fps={f}")
        for seq, f in cloud_dev_fps.items():
            print(f"  [CLOUD dev {seq}] inference fps={f}  (intermediate_queue batches only)")
        print(f"  [SYSTEM THROUGHPUT] {system_fps} fps  ({valid_batches} batches handled / wall-clock span)")
        print("=" * 50)
        Log.print_with_color(f"Saved {out_path} ({n_rows} batches)", "green")
        n_edge_devices = len(set(r.get("device_seq") for r in edge_rows))
        if n_edge_devices > 1:
            Log.print_with_color(
                f"[mAP] Skipped: {n_edge_devices} edge devices in this cluster — "
                f"frame alignment undefined for multi-edge mAP.", "yellow")
        elif self._map_updated:
            self._print_map()

        if self._det_results:
            self._write_detections_json()

    def inference_func(self, model, data, num_layers, splits, batch_size, logger, compress, mode="split", queue_name="intermediate_queue", save_set=None):
        if queue_name != self.intermediate_queue:
            self.intermediate_queue = queue_name
            self.channel.queue_declare(self.intermediate_queue, durable=False,
                                       arguments=get_intermediate_queue_args())

        if self.layer_id == 1:
            try:
                self.first_layer(model, data, batch_size, splits, logger, compress, mode, save_set)
            except Exception as e:
                Log.print_with_color(f"[!] Error during inference: {e!r} — saving metrics anyway.", "yellow")
                traceback.print_exc()
            if mode == "only_edge":
                self._pivot_and_save()
        elif self.layer_id == num_layers:
            self._setup_metrics_fanout_queue()
            try:
                self.last_layer(model, batch_size, splits, logger, compress, mode, save_set)
            except Exception as e:
                Log.print_with_color(f"[!] Error during inference: {e!r} — saving metrics anyway.", "yellow")
                traceback.print_exc()
            self._pivot_and_save()
        else:
            self.middle_layer(model)
