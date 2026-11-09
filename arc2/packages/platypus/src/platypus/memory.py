"""How much memory this process tree is actually using.

Summing RSS across a process tree badly over-counts. Every game child is
a ``spawn`` of the same interpreter and imports the same wheels, so a
resident vllm shows up in each one's RSS at full size while being a
single copy-on-write mapping of the same physical pages. A 32-way fan-out
reads as hundreds of gigabytes on a machine holding tens.

PSS (proportional set size, ``/proc/<pid>/smaps_rollup``) is the figure
that adds up: the kernel divides each page's size by the number of
processes mapping it, so a page shared by eight children contributes an
eighth to each, and the tree total counts it exactly once. That total is
the number to compare against a cgroup or SLURM memory limit.

Two properties worth knowing before reading the numbers:

- **A single process's PSS is not "its" memory.** It is that process's
  share of what it maps. Starting one more child lowers every sibling's
  PSS without anything being freed. The per-process figure that answers
  "what would be freed if this died" is USS (private pages only), and it
  is reported alongside for exactly that reason.
- **Pages shared with processes outside the tree** (a system libc, page
  cache) are divided by those sharers too, so the tree total slightly
  under-states what killing the tree would free. There is no way to have
  both this and no double counting; not double counting is the one that
  makes the total meaningful.

Linux-only, by construction — ``smaps_rollup`` is where the numbers live.
Elsewhere (and on kernels without it) the reader falls back to RSS and
says so, rather than reporting a silently wrong total.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_PROC = Path("/proc")


@dataclass(frozen=True)
class ProcessMemory:
    """One process's footprint, in kB (the units ``/proc`` reports)."""

    pid: int
    rss_kb: int
    #: Proportional share of everything mapped — the field that sums.
    #: ``None`` when the kernel does not expose ``smaps_rollup``.
    pss_kb: int | None
    #: Private (unshared) pages: what would be freed if this process died.
    uss_kb: int | None

    @property
    def countable_kb(self) -> int:
        """PSS where available, else RSS — so a fallback total is an
        over-count rather than a missing number."""
        return self.rss_kb if self.pss_kb is None else self.pss_kb


def _read_smaps_rollup(pid: int) -> tuple[int | None, int | None, int | None]:
    """``(rss, pss, uss)`` in kB from ``smaps_rollup``, or all-None."""
    try:
        text = (_PROC / str(pid) / "smaps_rollup").read_text()
    except (OSError, ValueError):
        return (None, None, None)
    rss = pss = private_clean = private_dirty = None
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key == "Rss":
            rss = int(rest.split()[0])
        elif key == "Pss":
            pss = int(rest.split()[0])
        elif key == "Private_Clean":
            private_clean = int(rest.split()[0])
        elif key == "Private_Dirty":
            private_dirty = int(rest.split()[0])
    uss = None if private_clean is None or private_dirty is None else private_clean + private_dirty
    return (rss, pss, uss)


def _read_statm_rss(pid: int) -> int | None:
    """RSS in kB from ``statm`` — the fallback when there is no rollup."""
    try:
        fields = (_PROC / str(pid) / "statm").read_text().split()
    except (OSError, ValueError):
        return None
    if len(fields) < 2:
        return None
    return int(fields[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)


def read_process_memory(pid: int) -> ProcessMemory | None:
    """One process's footprint, or ``None`` if it is gone or unreadable.

    A process exiting between listing and reading is routine at this
    cadence, not an error."""
    rss, pss, uss = _read_smaps_rollup(pid)
    if rss is None:
        rss = _read_statm_rss(pid)
    if rss is None:
        return None
    return ProcessMemory(pid=pid, rss_kb=rss, pss_kb=pss, uss_kb=uss)


def _parent_map() -> dict[int, int]:
    """``pid -> ppid`` for every visible process. Parsed from ``stat``,
    whose second field is the executable name in parentheses and may
    itself contain spaces and parentheses — so the tail is taken from the
    LAST ``)``, never split from the left."""
    parents: dict[int, int] = {}
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = (entry / "stat").read_text()
        except (OSError, ValueError):
            continue  # exited between listing and reading
        close = line.rfind(")")
        fields = line[close + 2 :].split()
        if len(fields) < 2:
            continue
        parents[int(entry.name)] = int(fields[1])
    return parents


def descendants(root_pid: int) -> list[int]:
    """Every process below ``root_pid``, at any depth.

    Grandchildren matter here and are easy to forget: a game child spawns
    kernel workers of its own, and the vLLM engine runs its core in a
    child of ours — all of them memory this run is responsible for."""
    children: dict[int, list[int]] = {}
    for pid, ppid in _parent_map().items():
        children.setdefault(ppid, []).append(pid)
    found: list[int] = []
    frontier = list(children.get(root_pid, ()))
    while frontier:
        pid = frontier.pop()
        found.append(pid)
        frontier.extend(children.get(pid, ()))
    return sorted(found)


@dataclass(frozen=True)
class TreeMemory:
    """A whole process tree's footprint: the root, its descendants, and
    the totals that do not double count."""

    root: ProcessMemory | None
    children: tuple[ProcessMemory, ...]

    @property
    def all_processes(self) -> tuple[ProcessMemory, ...]:
        """The tree: this process followed by its descendants."""
        return self.children if self.root is None else (self.root, *self.children)

    @property
    def total_kb(self) -> int:
        """The tree's real footprint — PSS summed, each physical page
        counted once."""
        return sum(process.countable_kb for process in self.all_processes)

    @property
    def children_total_kb(self) -> int:
        """The descendants' share of the tree total. Because PSS is additive
        this plus the root is the total EXACTLY — which is the point of
        printing it: the decomposition either accounts for the host
        process's footprint or names how much of it is the host itself."""
        return sum(process.countable_kb for process in self.children)

    @property
    def rss_sum_kb(self) -> int:
        """What naively adding up RSS would have said. Kept only so the
        difference (the shared part) can be shown: it is the number that
        makes the total look wrong when read on its own."""
        return sum(process.rss_kb for process in self.all_processes)

    @property
    def has_pss(self) -> bool:
        """Whether PSS was readable — it needs permission the RSS fallback does not."""
        return all(process.pss_kb is not None for process in self.all_processes)


def read_tree_memory(root_pid: int | None = None) -> TreeMemory:
    """Measure this process and everything below it.

    Cost is a page-table walk per process, so this belongs on a slow
    cadence (the run's periodic report), never in a hot path."""
    pid = os.getpid() if root_pid is None else root_pid
    measured = [read_process_memory(child) for child in descendants(pid)]
    return TreeMemory(
        root=read_process_memory(pid),
        children=tuple(process for process in measured if process is not None),
    )


# ---------------------------------------------------------------------------
# The cap we are actually judged against
# ---------------------------------------------------------------------------
#
# The tree total says what we use; it does not say what we are allowed. Under
# SLURM (and in the Kaggle container) that ceiling is a cgroup, and the number
# the kernel weighs against it is the cgroup's own — which is NOT our PSS
# total: it also carries page cache we caused, kernel allocations charged to
# us, and anything else in the cgroup. Reporting both, side by side, is what
# turns "we are at 61 G" into "we are at 71 of 120 G, and 61 of those 71 are
# ours" — the difference is the part an OOM kill would otherwise come out of
# nowhere from.

# cgroup v1 writes "no limit" as a page-counter maximum rather than a word
# (~9.2 exabytes). Compared in kB, which is what the reader has by then —
# a threshold in bytes silently never matches and the sentinel is reported
# as a real cap, leaving every run comfortably inside 8 exabytes.
_V1_UNLIMITED_ABOVE_KB = 1 << 50  # 1 EiB: no real cap is anywhere near this


@dataclass(frozen=True)
class CgroupMemory:
    """One cgroup's accounting, in kB. ``limit_kb`` is None when uncapped.

    The components come from ``memory.stat`` and are what make
    ``current_kb`` interpretable. Page cache is charged to whoever first
    touched the file and is mapped by no process, so it is invisible to
    PSS and is most of any gap between the tree total and ``current_kb`` —
    loading tens of gigabytes of model weights puts them here."""

    path: str
    current_kb: int
    peak_kb: int | None
    limit_kb: int | None
    anon_kb: int | None = None
    file_kb: int | None = None
    kernel_kb: int | None = None
    #: tmpfs pages. Charged inside ``file_kb`` by the kernel, but they have
    #: no backing file to be evicted to, so reclaiming them means swapping
    #: them out -- and a container with no swap cannot.
    shmem_kb: int | None = None
    #: The cgroup's swap ceiling. ``None`` covers both "unreadable" and
    #: "unlimited", which are indistinguishable here because ``max`` parses
    #: to None -- and both are read as "assume no swap", the pessimistic
    #: side: this figure predicts a kill, so overstating it costs a second
    #: look and understating it costs the job.
    swap_max_kb: int | None = None

    @property
    def unreclaimable_kb(self) -> int | None:
        """What a reclaim pass could NOT free, and therefore the figure
        that predicts an OOM kill.

        Hitting ``memory.max`` triggers reclaim before any kill, and page
        cache goes first. A cgroup sitting at 92% of its cap with a third
        of that in cache has real headroom; one at 92% with none has
        almost none. ``current`` alone cannot tell those apart, which is
        why it is the wrong number to panic on."""
        if self.anon_kb is None:
            return None
        total = self.anon_kb + (self.kernel_kb or 0)
        # tmpfs is charged as page cache and is not page cache: a host KV
        # tier, a /dev/shm cache, an engine's shared buffers all live here,
        # and without swap none of them can be reclaimed. Counting them as
        # cache is how a job reads as 5% spoken for while it is 41%.
        if self.shmem_kb and not self.swap_max_kb:
            total += self.shmem_kb
        return total


@dataclass(frozen=True)
class EnforcedLimit:
    """What bounds this run, and who set it."""

    cgroup: CgroupMemory | None
    slurm_job_id: str | None
    #: Cap SLURM was ASKED for, from the environment — the fallback when the
    #: cgroup files are not readable, and never preferred over them: the
    #: request is what we wanted, the cgroup is what we got.
    requested_kb: int | None


def _read_int(path: Path) -> int | None:
    """First integer in a sysfs/proc file, or None if unreadable."""
    try:
        text = path.read_text().strip()
    except (OSError, ValueError):
        return None
    if text == "max":
        return None
    try:
        return int(text) // 1024
    except ValueError:
        return None


def _read_stat(path: Path) -> dict[str, int]:
    """``memory.stat``'s ``key value`` lines, values converted to kB.
    Empty when unreadable — the breakdown is an enrichment, never a
    precondition for reporting the total."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return {}
    stat: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        try:
            stat[key] = int(value) // 1024
        except ValueError:
            continue
    return stat


def _own_cgroup_dir(proc_cgroup: Path, mount_root: Path) -> Path | None:
    """This process's memory cgroup directory, v2 or v1."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except (OSError, ValueError):
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and not controllers:  # unified (v2)
            candidate = mount_root / path.lstrip("/")
            if (candidate / "memory.current").exists():
                return candidate
        elif "memory" in controllers.split(","):  # v1
            candidate = mount_root / "memory" / path.lstrip("/")
            if (candidate / "memory.usage_in_bytes").exists():
                return candidate
    return None


def _job_cgroup_dir(own: Path) -> Path:
    """Climb to the whole job's cgroup where SLURM made one.

    Our own cgroup is the STEP. A job with several steps would report only
    this step's share, which is not what the job is judged on — so prefer
    the ``job_<id>`` ancestor when it is there and readable."""
    for ancestor in own.parents:
        if ancestor.name.startswith("job_") and (
            (ancestor / "memory.current").exists() or (ancestor / "memory.usage_in_bytes").exists()
        ):
            return ancestor
        if ancestor.name == "slurm" or ancestor == ancestor.parent:
            break
    return own


def job_cgroup_dir(
    proc_cgroup: Path = Path("/proc/self/cgroup"), mount_root: Path = Path("/sys/fs/cgroup")
) -> Path | None:
    """This run's cgroup directory -- the SLURM job's where there is one,
    else this process's own -- or None where there is none to read."""
    own = _own_cgroup_dir(proc_cgroup, mount_root)
    return None if own is None else _job_cgroup_dir(own)


def read_cgroup_memory(
    proc_cgroup: Path = Path("/proc/self/cgroup"), mount_root: Path = Path("/sys/fs/cgroup")
) -> CgroupMemory | None:
    """This run's cgroup accounting, or None where there is none to read."""
    own = _own_cgroup_dir(proc_cgroup, mount_root)
    if own is None:
        return None
    directory = _job_cgroup_dir(own)
    stat = _read_stat(directory / "memory.stat")
    if (directory / "memory.current").exists():  # v2
        current = _read_int(directory / "memory.current")
        peak = _read_int(directory / "memory.peak")
        limit = _read_int(directory / "memory.max")
        anon, file_, kernel = (stat.get(key) for key in ("anon", "file", "kernel"))
        shmem = stat.get("shmem")
        swap_max = _read_int(directory / "memory.swap.max")
    else:  # v1
        current = _read_int(directory / "memory.usage_in_bytes")
        peak = _read_int(directory / "memory.max_usage_in_bytes")
        limit = _read_int(directory / "memory.limit_in_bytes")
        if limit is not None and limit > _V1_UNLIMITED_ABOVE_KB:
            limit = None
        # v1 spells the same two things differently and has no kernel total.
        anon, file_, kernel = stat.get("rss"), stat.get("cache"), None
        # v1 reports neither under these names; unknown rather than absent.
        shmem, swap_max = None, None
    if current is None:
        return None
    return CgroupMemory(
        path=str(directory),
        current_kb=current,
        peak_kb=peak,
        limit_kb=limit,
        anon_kb=anon,
        file_kb=file_,
        kernel_kb=kernel,
        shmem_kb=shmem,
        swap_max_kb=swap_max,
    )


def _requested_slurm_kb(env: Mapping[str, str]) -> int | None:
    """The cap this job asked for, in kB, from SLURM's own environment."""

    def _int(name: str) -> int | None:
        """An integer environment value, or None when unset or unparsable."""
        try:
            return int(env[name])
        except (KeyError, ValueError):
            return None

    per_node = _int("SLURM_MEM_PER_NODE")  # SLURM reports these in MB
    if per_node:
        return per_node * 1024
    per_cpu, cpus = _int("SLURM_MEM_PER_CPU"), _int("SLURM_CPUS_ON_NODE")
    if per_cpu and cpus:
        return per_cpu * cpus * 1024
    per_gpu, gpus = _int("SLURM_MEM_PER_GPU"), _int("SLURM_GPUS_ON_NODE")
    if per_gpu and gpus:
        return per_gpu * gpus * 1024
    return None


def read_enforced_limit(environ: Mapping[str, str] | None = None, **cgroup_kwargs: Path) -> EnforcedLimit | None:
    """What this run is bounded by, or None when nothing worth reporting
    bounds it (no cgroup cap and no SLURM job — an ordinary dev box, where
    the line would be noise)."""
    env = os.environ if environ is None else environ
    job_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
    cgroup = read_cgroup_memory(**cgroup_kwargs)
    requested = _requested_slurm_kb(env) if job_id else None
    capped = cgroup is not None and cgroup.limit_kb is not None
    if not capped and job_id is None:
        return None
    return EnforcedLimit(cgroup=cgroup, slurm_job_id=job_id, requested_kb=requested)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _size(kb: float) -> str:
    """Kilobytes as a short human figure (M or G)."""
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.1f}G"
    if kb >= 1024:
        return f"{kb / 1024:.0f}M"
    return f"{kb:.0f}K"


def _percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile. Exact and obvious at the sizes here (tens
    of processes), where interpolation would only invent precision."""
    # ceil, not round: at an exact rank Python rounds half to even, which
    # moves the answer up or down depending on the parity of the rank.
    index = min(len(sorted_values) - 1, max(0, math.ceil(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def format_breakdown_line(cgroup: CgroupMemory) -> str | None:
    """What the cgroup's total is made of, and how much of it a reclaim
    pass could not free.

    This is what turns "72% of cap" into a decision. Page cache is charged
    to us because we read the model weights, is mapped by no process (so
    PSS cannot see it), and is dropped before the kernel kills anything —
    so a total dominated by it is not the emergency the percentage
    suggests."""
    if cgroup.anon_kb is None and cgroup.file_kb is None:
        return None
    parts: list[str] = []
    if cgroup.anon_kb is not None:
        parts.append(f"anon {_size(cgroup.anon_kb)}")
    if cgroup.shmem_kb:
        parts.append(f"tmpfs {_size(cgroup.shmem_kb)}")
    if cgroup.file_kb is not None:
        parts.append(f"page cache {_size(cgroup.file_kb)}")
    if cgroup.kernel_kb:
        parts.append(f"kernel {_size(cgroup.kernel_kb)}")
    line = "job memory: " + " + ".join(parts)
    unreclaimable = cgroup.unreclaimable_kb
    if unreclaimable is not None:
        line += f" | {_size(unreclaimable)} would survive reclaim"
        if cgroup.limit_kb:
            line += f" ({100.0 * unreclaimable / cgroup.limit_kb:.0f}% of cap)"
    return line


def format_limit_line(limit: EnforcedLimit, tree_total_kb: int) -> str:
    """The ceiling line: what we are held to, what the enforcer says we are
    using, and how much of that our own tree accounts for.

    The last part is the one that pays off during an incident. The cgroup
    figure carries page cache and kernel memory we caused but do not hold in
    any process, so it sits above the tree total; seeing the gap named makes
    an approaching OOM legible instead of contradictory."""
    who = f"slurm job {limit.slurm_job_id}" if limit.slurm_job_id else "cgroup"
    cgroup = limit.cgroup
    if cgroup is None:
        cap = f"{_size(limit.requested_kb)} cap requested" if limit.requested_kb else "cap unknown"
        return f"{who}: {cap} (cgroup not readable — tree pss {_size(tree_total_kb)} is all we can see)"

    parts = [f"{who}: holding {_size(cgroup.current_kb)}"]
    if cgroup.limit_kb is not None:
        parts.append(f"of {_size(cgroup.limit_kb)} cap ({100.0 * cgroup.current_kb / cgroup.limit_kb:.0f}%)")
    elif limit.requested_kb:
        parts.append(f"({_size(limit.requested_kb)} requested; cgroup uncapped)")
    else:
        parts.append("(uncapped)")
    if cgroup.peak_kb is not None:
        parts.append(f"peak {_size(cgroup.peak_kb)}")
    parts.append(f"| our tree accounts for {_size(tree_total_kb)}")
    return " ".join(parts)


def format_memory_lines(tree: TreeMemory, limit: EnforcedLimit | None = None) -> list[str]:
    """The report: the tree total with the shared part called out, then
    the spread across children.

    The shared figure is on the line deliberately. Anyone who has looked
    at these processes individually has seen ~5 GB each and will read a
    much smaller total as wrong; naming how much is shared is what makes
    the total believable."""
    processes = tree.all_processes
    if not processes:
        return []
    total = tree.total_kb
    metric = "pss" if tree.has_pss else "rss (no smaps_rollup — shared pages ARE double counted)"
    line = f"tree {_size(total)} {metric}, {len(processes)} procs"
    if tree.root is not None:
        # Split the total, not just the root: the question this answers is
        # whether the children explain the host process's footprint, and
        # PSS being additive means these three numbers must reconcile.
        line += f" = main {_size(tree.root.countable_kb)} + children {_size(tree.children_total_kb)}"
    if tree.has_pss:
        shared = tree.rss_sum_kb - total
        line += f" (rss sums to {_size(tree.rss_sum_kb)}; {_size(shared)} of that is shared)"
    lines = [line]

    if tree.children:
        values = sorted(process.countable_kb for process in tree.children)
        child_line = (
            f"{len(values)} children: min {_size(values[0])} med {_size(_percentile(values, 0.5))} "
            f"p90 {_size(_percentile(values, 0.9))} max {_size(values[-1])}"
        )
        # A child's own PSS falls as siblings start, purely from sharing;
        # the private figure is the one that answers "what is this process
        # actually costing us".
        private = sorted(p.uss_kb for p in tree.children if p.uss_kb is not None)
        if len(private) == len(values):
            child_line += f" | private max {_size(private[-1])}"
        lines.append(child_line)
    if limit is not None:
        lines.append(format_limit_line(limit, total))
        if limit.cgroup is not None:
            breakdown = format_breakdown_line(limit.cgroup)
            if breakdown is not None:
                lines.append(breakdown)
    return lines
