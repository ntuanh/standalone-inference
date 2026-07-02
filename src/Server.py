import numpy as np
import os
import sys
import glob
import base64
import time
import pika
import pickle
import src.Model
import src.Log
from src.Utils import get_intermediate_queue_args, get_bbox_queue_args
from ultralytics import YOLO

from src.Clustering import (
    ManualExperimentConfig,
    DeterministicSimilarityAssignmentSolver,
    run_manual_hungarian_case,
    print_result,
    get_cut_data_sizes,
    get_raw_input_mb,
)

class Server:
    def __init__(self, config):
        # One-time cleanup of shared metrics/lock files from a previous run.
        # Must happen here (server starts once) — doing this in each Scheduler
        # caused later-starting clients to wipe out files already being
        # written by clients that started earlier.
        for f in (
            glob.glob("metrics_raw_*.csv")
            + glob.glob("metrics_pivoted_*.csv")
            + glob.glob("metrics_pivot_*.lock")
            + ["detections_stream.jsonl"]
        ):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    src.Log.print_with_color(f"[!] Cannot delete {f} (file is open). Close it and retry.", "red")

        self.config = config
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        self.virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_size = config["server"]["batch-size"]

        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.address,
                port=5672,
                virtual_host=f"{self.virtual_host}",
                credentials=credentials,
                heartbeat=0,                # disable heartbeats (long inference
                                            # blocks the pika thread for seconds)
                blocked_connection_timeout=600
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue', durable=False)
        self.channel.queue_purge(queue='rpc_queue')

        # Discard any messages left over from a previous (crashed) run, so
        # depth-based back-pressure starts from an empty queue instead of
        # being thrown off by stale large messages still sitting in RabbitMQ.
        self.channel.queue_declare(queue='intermediate_queue', durable=False,
                                   arguments=get_intermediate_queue_args(config))
        self.channel.queue_purge(queue='intermediate_queue')

        # adaptive mode: edge ships locally-computed bboxes here (no cloud YOLO).
        # Kept separate from intermediate_queue so its depth doesn't pollute the
        # edge's "is the cloud backed up?" routing signal.
        self.channel.queue_declare(queue='bbox_queue', durable=False,
                                   arguments=get_bbox_queue_args(config))
        self.channel.queue_purge(queue='bbox_queue')

        # fps_queue: every cloud publishes one tiny "done" ping per completed
        # batch; the server consumes them (on_fps_done) and computes FPS live.
        self.channel.queue_declare(queue='fps_queue', durable=False)
        self.channel.queue_purge(queue='fps_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.registered_ids = set()
        self.notified = False
        self.count_clients = 0
        self.client_assignments = {}    # {client_id: {"splits": int, "queue_name": str}}
        self.client_profile_data = {}   # {client_id_str: np.array of per-layer times}
        self.client_bandwidth_data = {} # {client_id_str: float MB/s}
        self.client_name_data = {}      # {client_id_str: str name}
        self._stopping = False
        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        # FPS tracking state, fed by bare "DONE" pings on fps_queue — one per
        # batch, sent by whichever tier (edge or cloud) completed it. Every
        # arrival timestamp is recorded (server's own clock, so device clock
        # skew cannot distort it) and all FPS numbers derive from that list.
        # Exact system FPS = frames / wall-clock time. The arithmetic mean of
        # instantaneous 1/delta values is NOT used: bursty arrivals (tiny
        # deltas → huge fps entries) inflate it far above the real rate.
        self._done_times = []         # arrival time of every DONE
        self._run_start_ts = None     # when START was broadcast to clients
        # Queues that hold not-yet-processed batches. The post-STOP drain loop
        # keeps collecting DONEs while any of these are non-empty, so the server
        # cannot shut down before the clouds finish the backlog.
        self._work_queues = ["intermediate_queue", "bbox_queue"]
        self.channel.basic_consume(queue='fps_queue', on_message_callback=self.on_fps_done)

        self.data = config["data"]
        self.compress = config["compress"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", config["debug-mode"])
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def _get_mode(self):
        exp = self.config.get("experiment", {})
        if exp.get("enable", True):
            return exp.get("mode", "split")
        return "split"

    def on_request(self, ch, method, _, body):
        message = pickle.loads(body)
        action = message["action"]

        if action == "REGISTER":
            client_id = message["client_id"]
            layer_id = message["layer_id"]

            src.Log.print_with_color(f"[<<<] Received REGISTER from client {client_id} layer={layer_id}", "blue")

            if layer_id < 1 or layer_id > len(self.register_clients):
                src.Log.print_with_color(
                    f"[!] Ignored client with unexpected layer_id={layer_id} (expected 1..{len(self.register_clients)})", "red")
                return

            if str(client_id) in self.registered_ids:
                src.Log.print_with_color(f"[!] Duplicate REGISTER from {client_id}, ignored.", "yellow")
                return

            self.registered_ids.add(str(client_id))
            self.list_clients.append((str(client_id), layer_id))

            layer_times = message.get("layer_times", None)
            if layer_times is not None:
                self.client_profile_data[str(client_id)] = np.array(layer_times, dtype=float)
                src.Log.print_with_color(
                    f"[Profile] Stored profiling data from client {client_id} "
                    f"({len(layer_times)} layers, total={sum(layer_times)*1000:.1f} ms)", "cyan")

            bandwidth_mb_s = message.get("bandwidth_mb_s", None)
            if bandwidth_mb_s is not None:
                self.client_bandwidth_data[str(client_id)] = float(bandwidth_mb_s)
                src.Log.print_with_color(
                    f"[Bandwidth] Stored bandwidth from client {client_id}: {bandwidth_mb_s:.1f} MB/s", "cyan")

            client_name = message.get("client_name", None)
            if client_name:
                self.client_name_data[str(client_id)] = client_name

            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients and not self.notified:
                self.notified = True
                src.Log.print_with_color("All clients connected. Sending notifications.", "green")
                self.notify_clients()

        elif action == "BW_TEST":
            client_id = message["client_id"]
            self.send_to_response(str(client_id), pickle.dumps({"action": "BW_ACK"}))

        elif action == "NOTIFY":
            self.count_clients += 1
            if self.count_clients == self.total_clients[0]:
                self.logger.log_info("Stop Inference !!!")
                self._stopping = True
                self.notify_clients(start=False)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.channel.stop_consuming()
                return

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(exchange='', routing_key=reply_queue_name, body=message)

    def _handle_fps_done(self):
        """One batch (batch_size frames) fully handled somewhere in the system —
        whichever tier (edge or cloud) completed it published a bare "DONE".
        Record the arrival time and report three views of FPS:
          inst    = batch_size / delta since the previous DONE (noisy, bursty)
          window  = frames / time over the last <=16 DONEs (smoothed live view)
          system  = frames / time since the first DONE (exact cumulative rate)
        Only 'system' (and TOTAL TIME in the summary) is exact; inst/window
        are for watching the run live."""
        now = time.time()
        self._done_times.append(now)
        n = len(self._done_times)
        if n == 1:
            src.Log.print_with_color("[FPS] #1 DONE — timing started", "cyan")
            return

        delta = now - self._done_times[-2]
        inst_fps = self.batch_size / delta if delta > 0 else 0.0

        w = self._done_times[-16:]
        win_span = w[-1] - w[0]
        win_fps = (len(w) - 1) * self.batch_size / win_span if win_span > 0 else 0.0

        span = now - self._done_times[0]
        system_fps = (n - 1) * self.batch_size / span if span > 0 else 0.0

        src.Log.print_with_color(
            f"[FPS] #{n} DONE delta={delta:.3f}s inst={inst_fps:.2f} "
            f"| window_fps={win_fps:.2f} | system_fps={system_fps:.2f}", "cyan")

    def on_fps_done(self, ch, method, _, body):
        self._handle_fps_done()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _queue_depth(self, qname):
        """Ready-message count via passive declare. Only call on queues this
        server itself declared — a passive declare on a missing queue closes
        the channel (404)."""
        try:
            return self.channel.queue_declare(queue=qname, passive=True).method.message_count
        except Exception:
            return 0

    def _drain_fps_pings(self, idle_grace_s=10.0):
        """The STOP broadcast fires when all EDGES notify — the clouds are
        usually still chewing through the backlog at that point, so the server
        must NOT shut down yet or it loses their remaining DONEs. Keep
        collecting until (a) every work queue is empty AND (b) no DONE arrived
        for idle_grace_s. The grace period covers the final batches: a batch
        being processed right now is already off the queue (auto_ack), so it is
        invisible to the depth check until its DONE lands."""
        last_msg = time.time()
        while True:
            try:
                method_frame, _, body = self.channel.basic_get(queue='fps_queue', auto_ack=True)
            except Exception:
                return
            if method_frame:
                self._handle_fps_done()
                last_msg = time.time()
                continue
            backlog = sum(self._queue_depth(q) for q in self._work_queues)
            if backlog > 0:
                last_msg = time.time()  # clouds still have queued work — keep waiting
            elif time.time() - last_msg >= idle_grace_s:
                return
            time.sleep(0.2)

    def _print_fps_summary(self):
        """Exact system FPS = frames / wall-clock time. Two anchors are shown:
        START→last DONE (whole run incl. warm-up: model load + first batch) and
        first→last DONE (steady state). The arithmetic mean of per-DONE 1/delta
        values is printed only as a reference — bursty arrivals make it read
        far above the rate the system actually sustained."""
        n = len(self._done_times)
        if n < 2:
            src.Log.print_with_color(
                "[FPS] Fewer than 2 'DONE' pings received — cannot compute FPS.", "yellow")
            return
        t_first, t_last = self._done_times[0], self._done_times[-1]
        frames = n * self.batch_size
        span = t_last - t_first
        steady_fps = (n - 1) * self.batch_size / span if span > 0 else 0.0
        deltas = [b - a for a, b in zip(self._done_times, self._done_times[1:])]
        inst = [self.batch_size / d for d in deltas if d > 0]
        arith_mean = sum(inst) / len(inst) if inst else 0.0

        print("=" * 60)
        print(f"  [FPS SUMMARY]  batches={n}  frames={frames}")
        if self._run_start_ts is not None:
            total = t_last - self._run_start_ts
            total_fps = frames / total if total > 0 else 0.0
            print(f"  TOTAL TIME (START -> last DONE) = {total:.2f}s")
            print(f"  SYSTEM FPS (frames/total time)  = {total_fps:.3f}")
        print(f"  first->last DONE span = {span:.2f}s  -> steady-state FPS = {steady_fps:.3f}")
        print(f"  (mean of per-DONE 1/delta fps = {arith_mean:.3f} — inflated by bursts, reference only)")
        print("=" * 60)

    def start(self):
        self.channel.start_consuming()
        # STOP has been broadcast, but clouds may still be finishing their
        # backlog — keep collecting fps pings until the queue goes quiet,
        # then print the final system FPS.
        self._drain_fps_pings()
        self._print_fps_summary()
        self.connection.close()
        sys.exit(0)

    def _run_hungarian(self):
        cfg = self.config.get("clustering", {})
        network_rate = float(cfg.get("network_rate_mb_s", 1000.0))
        max_clusters = cfg.get("max_clusters", 1)

        # Dùng real profiling data nếu tất cả client đã gửi
        edge_times_list = [
            self.client_profile_data[str(cid)]
            for cid, lid in self.list_clients
            if lid == 1 and str(cid) in self.client_profile_data
        ]
        cloud_times_list = [
            self.client_profile_data[str(cid)]
            for cid, lid in self.list_clients
            if lid == len(self.total_clients) and str(cid) in self.client_profile_data
        ]
        n_edge = sum(1 for _, lid in self.list_clients if lid == 1)
        n_cloud = sum(1 for _, lid in self.list_clients if lid == len(self.total_clients))
        profile_source = cfg.get("profile_source", "auto")
        has_real = (len(edge_times_list) == n_edge and len(cloud_times_list) == n_cloud
                    and n_edge > 0 and n_cloud > 0)

        if profile_source == "real" and not has_real:
            raise RuntimeError("[Clustering] profile_source=real nhưng chưa có đủ profiling từ clients")

        use_real = has_real if profile_source == "auto" else (profile_source == "real")

        if use_real:
            src.Log.print_with_color(
                f"[Clustering] Using REAL profiles ({n_edge} edge, {n_cloud} cloud) [profile_source={profile_source}]", "cyan")
            N = len(edge_times_list)
            M = len(cloud_times_list)
            edge_clients = [cid for cid, lid in self.list_clients
                            if lid == 1 and str(cid) in self.client_profile_data]
            rates_matrix = np.array([
                [self.client_bandwidth_data.get(str(cid), network_rate)] * M
                for cid in edge_clients
            ]) if edge_clients else np.full((N, M), network_rate)
            cloud_clients = [cid for cid, lid in self.list_clients
                             if lid == len(self.total_clients) and str(cid) in self.client_profile_data]
            solver = DeterministicSimilarityAssignmentSolver(
                client_layer_times=np.vstack(edge_times_list),
                server_layer_times=np.vstack(cloud_times_list),
                cut_data_sizes=get_cut_data_sizes(self.model_name, self.batch_size),
                input_data_size=get_raw_input_mb(self.batch_size),
                network_rates=rates_matrix,
            )
            solver.client_type_names = [
                self.client_name_data.get(str(cid), f"edge_{str(cid)[:8]}")
                for cid in edge_clients
            ]
            solver.cloud_type_names = [
                self.client_name_data.get(str(cid), f"cloud_{str(cid)[:8]}")
                for cid in cloud_clients
            ]
            result = solver.solve_best_over_k("hungarian", max_clusters=max_clusters)["best_result"]
            print_result(result, solver, title="HUNGARIAN MATCHING RESULT (real profiles)")
        else:
            src.Log.print_with_color(
                f"[Clustering] Using SIMULATED profiles (DEVICE_A/B/C hardcoded) [profile_source={profile_source}]", "yellow")
            manual_cfg = ManualExperimentConfig(
                num_A=cfg.get("num_A", 1),
                num_B=cfg.get("num_B", 0),
                num_C=cfg.get("num_C", 0),
                num_cloud=cfg.get("num_cloud", 1),
                network_rate_mb_s=network_rate,
                max_clusters=max_clusters,
                exact_max_k=max_clusters,
                model_name=self.model_name,
                batch_size=self.batch_size,
                input_data_mb=get_raw_input_mb(self.batch_size),
            )
            results = run_manual_hungarian_case(manual_cfg)
            solver = results["solver"]
            result = results["hungarian"]

        return solver, result

    def notify_clients(self, start=True):
        if start:
            default_splits = {"a": 4, "b": 11, "c": 17, "d": 23}

            if os.path.exists(f"{self.model_name}.pt"):
                src.Log.print_with_color(f"Exist {self.model_name}.pt", "green")
            else:
                src.Log.print_with_color(f"Download {self.model_name}", "yellow")
                _ = YOLO(f"{self.model_name}.pt")

            mode = self._get_mode()
            splits = None

            if mode in ["only_edge", "only_cloud", "adaptive"]:
                src.Log.print_with_color(f"[Benchmark] mode={mode}, skip split selection", "yellow")

            else:
                clustering_cfg = self.config.get("clustering", {})
                use_hungarian = clustering_cfg.get("enable", False)

                if use_hungarian:
                    try:
                        _, h = self._run_hungarian()
                        edge_labels  = h.edge_labels
                        cloud_labels = h.cloud_labels
                        matching     = h.matching
                        best_cuts    = h.best_cuts
                        K            = h.num_clusters
                        inv_matching = {int(matching[k]): k for k in range(K)}

                        edge_ord  = [(cid, lid) for cid, lid in self.list_clients if lid == 1]
                        cloud_ord = [(cid, lid) for cid, lid in self.list_clients if lid == len(self.total_clients)]

                        self.client_assignments = {}
                        for i, (cid, _) in enumerate(edge_ord):
                            k = int(edge_labels[i]) if i < len(edge_labels) else 0
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }
                        for j, (cid, _) in enumerate(cloud_ord):
                            l = int(cloud_labels[j]) if j < len(cloud_labels) else 0
                            k = inv_matching.get(l, 0)
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }

                        splits = int(best_cuts[0]) + 1 if len(best_cuts) > 0 else None
                        src.Log.print_with_color(
                            f"[Clustering] K={K}  best_cuts={best_cuts.tolist()}", "green")

                        # Discard leftovers from a previous (crashed) run for each
                        # per-cluster queue, same reasoning as 'intermediate_queue' above.
                        for k in range(K):
                            qname = f"intermediate_queue_{k}"
                            self.channel.queue_declare(queue=qname, durable=False,
                                                       arguments=get_intermediate_queue_args(self.config))
                            self.channel.queue_purge(queue=qname)
                            # watched by the post-STOP FPS drain loop
                            if qname not in self._work_queues:
                                self._work_queues.append(qname)

                    except Exception as e:
                        raise RuntimeError(f"Hungarian clustering failed: {e}")

                elif self.cut_layer in default_splits:
                    splits = default_splits[self.cut_layer]
                    src.Log.print_with_color(
                        f"[Benchmark] Fixed split '{self.cut_layer}' -> splits={splits}", "yellow")
                else:
                    raise ValueError(f"Invalid cut-layer: '{self.cut_layer}'. Use a/b/c/d or set clustering.enable: True")

            file_path = f"{self.model_name}.pt"
            if not os.path.exists(file_path):
                src.Log.print_with_color(f"{self.model_name}.pt does not exist.", "yellow")
                self.connection.close()
                sys.exit(1)

            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')

            # Deduplicate list_clients trong trường hợp pika callback reentrant
            seen_notify = set()
            clients_to_notify = []
            for entry in self.list_clients:
                if entry[0] not in seen_notify:
                    seen_notify.add(entry[0])
                    clients_to_notify.append(entry)

            src.Log.print_with_color(
                f"Sending model {self.model_name} to {len(clients_to_notify)} clients "
                f"(list_clients={len(self.list_clients)}).", "green")

            for (client_id, layer_id) in clients_to_notify:
                assignment = self.client_assignments.get(client_id, {})
                response = {
                    "action":     "START",
                    "message":    "Server accept the connection",
                    "model":      encoded,
                    "splits":     assignment.get("splits",     splits),
                    "queue_name": assignment.get("queue_name", "intermediate_queue"),
                    "batch_size": self.batch_size,
                    "num_layers": len(self.total_clients),
                    "model_name": self.model_name,
                    "data":       self.data,
                    "compress":   self.compress,
                    "mode":       self._get_mode(),
                }
                self.send_to_response(client_id, pickle.dumps(response))

            # Inference effectively starts now — anchor for TOTAL TIME and the
            # exact SYSTEM FPS (frames / total time) in the final summary.
            self._run_start_ts = time.time()
        else:
            response = {"action": "STOP", "message": "Stop inference !!!"}
            for (client_id, layer_id) in self.list_clients:
                self.send_to_response(client_id, pickle.dumps(response))
