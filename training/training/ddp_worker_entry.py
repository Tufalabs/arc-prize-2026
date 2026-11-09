"""Protect a worker from group SIGTERM before importing Torch or loading models."""

import os

from training.shutdown import worker_sigterm_handler


def main() -> None:
    with worker_sigterm_handler(os.getppid()):
        from training.ddp_worker import main as run

        run()


if __name__ == "__main__":
    main()
