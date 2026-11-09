"""Build a parallel TensorBoard index whose RELATIVE axis counts only
active training time: every run linked in --src is rewritten into --dst
with queue-wait gaps collapsed (see training/tb_gap_collapse.py, which
the trainer also applies in place at each resume). Idempotent full
regeneration per pass; loop it to track live runs. Originals untouched.

    python scripts/rebase_tensorboard_active_time.py --src ~/tb_20260826 --dst ~/tb_active
    tensorboard --logdir ~/tb_active ...
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.tb_gap_collapse import DEFAULT_GAP_THRESHOLD_SECONDS, collapse_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="~/tb_20260826")
    parser.add_argument("--dst", default="~/tb_active")
    parser.add_argument(
        "--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD_SECONDS
    )
    args = parser.parse_args()
    src_index = os.path.expanduser(args.src)
    dst_index = os.path.expanduser(args.dst)
    os.makedirs(dst_index, exist_ok=True)
    kept = set()
    for name in sorted(os.listdir(src_index)):
        src = os.path.realpath(os.path.join(src_index, name))
        if not os.path.isdir(src):
            continue
        collapsed = collapse_run_dir(
            src,
            gap_threshold=args.gap_threshold,
            output_dir=os.path.join(dst_index, name),
        )
        kept.add(name)
        print("%-24s gaps_collapsed=%d" % (name, collapsed))
    for name in os.listdir(dst_index):
        if name not in kept and not name.endswith(".gap-staging"):
            shutil.rmtree(os.path.join(dst_index, name), ignore_errors=True)


if __name__ == "__main__":
    main()
