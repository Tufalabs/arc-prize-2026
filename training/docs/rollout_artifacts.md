# Retaining rollout artifacts

With `training.cleanup_rollout_artifacts=false` and node-local rollout
scratch configured, the trainer queues each window after optimization and
viewer export have finished reading its source files. Training does not
wait for archiving or transfer. Its rollout schedule, samples, policy
publication, metrics and checkpoint cadence are unchanged.

One separate Python process at a time packs the window into a gzip archive
on local scratch, then streams that one file to shared storage. The worker
uses only the Python standard library, hides CUDA devices, limits itself to
one CPU from the allocation's existing affinity, and lowers its CPU priority
to nice 19. When available, `ionice` also sets idle I/O priority. Compression
uses gzip level 1. It does not import torch or initialize CUDA, and packing
and copying do not hold the trainer's Python GIL.
Archive diagnostics use a separate stderr handler so they never acquire the
trainer's shared NFS file-log handler lock.

NFS transfers are paced in 256 KiB chunks, with a default limit of **8 MiB/s
per trainer**, including shutdown. There is one archive writer and a queue
of at most eight pending windows. A full queue leaves further windows on
scratch for the shutdown sweep. Transfer failures retain the source and
receive one background retry after five seconds; shutdown retries any data
still on scratch. No file-by-file NFS copy runs alongside the archiver.
Limits are per trainer, so concurrent jobs' bandwidth adds up.

The worker flushes the completed remote file, atomically renames its hidden
`.partial` file, and flushes the containing directory before deleting the
source window. Unique names preserve prior segments' archives. A failed or
interrupted copy cannot appear under a completed archive's name. On Linux,
the writer also exits if its owning trainer dies.

After collection stops, shutdown drains the queue and archives residual
rollout roots (including incomplete windows and the final retained official
evaluation artifacts). The default drain budget is **120 seconds**. If the
budget expires, the writer is killed and an error identifies the remaining
scratch paths; the trainer never starts an unbounded copy fallback.
The parent tracks the local archive and remote partial file and reclaims
both in the background after a killed worker is confirmed reaped. An
unreaped writer retains ownership of its temporary files; failed cleanup
operations are logged without masking the original archive error.
Server logs are still staged. Existing exception handling preserves an original
training failure or scheduled shutdown instead of masking it with a
secondary archive error. Scheduler SIGKILL/node loss can prevent this final
drain, and job-local scratch may then be reclaimed. A prolonged NFS outage
can also grow the scratch backlog; inspect archive warnings and free space.

Operators can tune finite positive limits before submitting the job:

```bash
export ARC3_ROLLOUT_ARCHIVE_MIB_PER_SECOND=8
export ARC3_ROLLOUT_ARCHIVE_SHUTDOWN_SECONDS=120
```

Per-window archives live at
`<run_root>/rollout_archives/rollouts/window_<id>.<unique_id>.tar.gz` and
contain the `window_<id>/` tree. For example:

```bash
mkdir -p "$RUN_DIR/rollouts"
tar -xzf "$WINDOW_ARCHIVE" -C "$RUN_DIR/rollouts"
```

The shutdown sweep writes `remaining.<unique_id>.tar.gz` in the rollout
root's archive directory. These archives contain the root directory itself
(`rollouts/` or `official_eval_rollouts/`); extract them into `$RUN_DIR`.
Archives are debugging artifacts and are not loaded by training resume.

When cleanup is enabled, or rollout files already live in the durable run
tree, the existing behavior applies and no background archiver is started.
The dedicated official-evaluation worker keeps its existing latest-cycle
retention policy. This change addresses artifact stage-out in issue #102;
the speculative window discarded at segment boundaries is a separate
collector scheduling change.
