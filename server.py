import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues, acquire_server_lock
import src.Log
import yaml

parser = argparse.ArgumentParser(description="Split learning framework with controller.")
args = parser.parse_args()

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # Before delete_old_queues, which is destructive: it deletes the work queues,
    # the permit pools and rpc_queue, so bailing out after it would already have
    # wrecked a run in progress.
    try:
        # Held for the whole process lifetime — dropping this reference releases
        # the lock and lets a second server in.
        server_lock = acquire_server_lock(address, username, password, virtual_host)
    except RuntimeError as e:
        src.Log.print_with_color(f"[!] {e}", "red")
        sys.exit(1)

    delete_old_queues(address, username, password, virtual_host)
    server = Server(config)
    server.start()
