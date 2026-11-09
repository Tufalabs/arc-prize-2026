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

import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

_PROC = Path("/proc")

# What a process in the tree is, for the report. A run holds one TAAF host
# process, one vLLM engine (a core plus its workers), one small process per
# hosted game and one REPL per game kernel; a single number for "children"
# hides which of those is growing.
TAAF = "TAAF"
VLLM = "vLLM"
GAME = "game"
REPL = "REPL"
OTHER = "other"

# vLLM retitles every process it starts (``VLLM::EngineCore``,
# ``VLLM::Worker_PP0_TP0``) under this prefix, VLLM_PROCESS_NAME_PREFIX's
# default; the title is what /proc/<pid>/cmdline then reads.
_VLLM_TITLE_PREFIX = "VLLM::"

# pid -> game id for the live game subprocesses of THIS process, written by
# ``platypus.subproc`` as it starts and reaps them. Structure alone cannot
# name a game: every game child is the same spawned interpreter.
_game_processes: dict[int, str] = {}


def register_game_process(pid: Optional[int], game_id: str) -> None:
    """Name a game subprocess for the memory report. Its descendants are
    reported as that game's REPL."""
    if pid is not None:
        _game_processes[pid] = game_id


def forget_game_process(pid: Optional[int]) -> None:
    """Drop a reaped game subprocess, so a recycled pid is never misnamed."""
    if pid is not None:
        _game_processes.pop(pid, None)


@dataclass(frozen=True)
class ProcessMemory:
    """One process's footprint, in kB (the units ``/proc`` reports)."""

    pid: int
    rss_kb: int
    #: Proportional share of everything mapped — the field that sums.
    #: ``None`` when the kernel does not expose ``smaps_rollup``.
    pss_kb: Optional[int]
    #: Private (unshared) pages: what would be freed if this process died.
    uss_kb: Optional[int]
    #: What the process is (TAAF, VLLM, GAME, REPL, OTHER), set by the tree reader.
    role: str = OTHER
    #: Who it is within its role: the game id for a game or REPL, the
    #: executable name otherwise.
    name: str = ""
    #: Resident peak (``VmHWM``). Next to the current figure it says whether a
    #: process is holding its high-water mark or has come down from it.
    peak_kb: Optional[int] = None
    #: Anonymous resident memory (``RssAnon``): the heap and private maps,
    #: without shared libraries or tmpfs — what a leak grows.
    anon_kb: Optional[int] = None

    @property
    def countable_kb(self) -> int:
        """PSS where available, else RSS — so a fallback total is an
        over-count rather than a missing number."""
        return self.rss_kb if self.pss_kb is None else self.pss_kb


def _read_smaps_rollup(pid: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
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


def _read_statm_rss(pid: int) -> Optional[int]:
    """RSS in kB from ``statm`` — the fallback when there is no rollup."""
    try:
        fields = (_PROC / str(pid) / "statm").read_text().split()
    except (OSError, ValueError):
        return None
    if len(fields) < 2:
        return None
    return int(fields[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)


def _read_status_kb(pid: int) -> dict[str, int]:
    """``VmHWM`` and ``RssAnon`` from ``/proc/<pid>/status``, in kB; empty when
    unreadable."""
    found: dict[str, int] = {}
    try:
        text = (_PROC / str(pid) / "status").read_text()
    except (OSError, ValueError):
        return found
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmHWM", "RssAnon"):
            found[key] = int(rest.split()[0])
    return found


def read_process_memory(pid: int) -> Optional[ProcessMemory]:
    """One process's footprint, or ``None`` if it is gone or unreadable.

    A process exiting between listing and reading is routine at this
    cadence, not an error."""
    rss, pss, uss = _read_smaps_rollup(pid)
    if rss is None:
        rss = _read_statm_rss(pid)
    if rss is None:
        return None
    status = _read_status_kb(pid)
    return ProcessMemory(
        pid=pid, rss_kb=rss, pss_kb=pss, uss_kb=uss, peak_kb=status.get("VmHWM"), anon_kb=status.get("RssAnon")
    )


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


def _children_map() -> dict[int, list[int]]:
    """``ppid -> [pid, ...]`` for every visible process."""
    children: dict[int, list[int]] = {}
    for pid, ppid in _parent_map().items():
        children.setdefault(ppid, []).append(pid)
    return children


def descendants(root_pid: int) -> list[int]:
    """Every process below ``root_pid``, at any depth.

    Grandchildren matter here and are easy to forget: a game child spawns
    kernel workers of its own, and the vLLM engine runs its core in a
    child of ours — all of them memory this run is responsible for."""
    return sorted(_classify(root_pid, _children_map(), {}))


def _cmdline(pid: int) -> list[str]:
    """``/proc/<pid>/cmdline`` as words; empty when unreadable."""
    try:
        raw = (_PROC / str(pid) / "cmdline").read_bytes()
    except (OSError, ValueError):
        return []
    return [word.decode(errors="replace").strip() for word in raw.split(b"\0") if word]


def _process_title(pid: int) -> str:
    """The first word of the command line: the title a process set for
    itself, or its executable."""
    words = _cmdline(pid)
    return words[0] if words else ""


def _other_name(pid: int) -> str:
    """A short name for a process of no known role: its executable, and for
    an interpreter what it runs — ``python`` alone names nothing."""
    words = _cmdline(pid)
    if not words:
        return str(pid)
    name = Path(words[0]).name
    if name.startswith("python") and len(words) > 1:
        what = words[2] if words[1] in ("-c", "-m") and len(words) > 2 else words[1]
        name += " " + (what[:48] if words[1] == "-c" else Path(what).name)
    return name


def _classify(root_pid: int, children: Mapping[int, list[int]], games: Mapping[int, str]) -> dict[int, tuple[str, str]]:
    """``pid -> (role, name)`` for every process below ``root_pid``.

    A role is inherited down the tree: everything below a game process is
    its REPL (the kernel, and anything that kernel starts), everything below
    a vLLM-titled process is vLLM (workers, compile helpers)."""
    found: dict[int, tuple[str, str]] = {}
    frontier: list[tuple[int, Optional[tuple[str, str]]]] = [(pid, None) for pid in children.get(root_pid, ())]
    while frontier:
        pid, inherited = frontier.pop()
        if inherited is not None and inherited[0] == VLLM:
            # vLLM titles its workers too; keep their own names where set.
            title = _process_title(pid)
            role = (VLLM, title[len(_VLLM_TITLE_PREFIX) :]) if title.startswith(_VLLM_TITLE_PREFIX) else inherited
        elif inherited is not None:
            role = (REPL, inherited[1])
        elif pid in games:
            role = (GAME, games[pid])
        else:
            title = _process_title(pid)
            if title.startswith(_VLLM_TITLE_PREFIX):
                role = (VLLM, title[len(_VLLM_TITLE_PREFIX) :])
            else:
                role = (OTHER, _other_name(pid))
        found[pid] = role
        passed_down = role if role[0] in (GAME, REPL, VLLM) else None
        frontier.extend((child, passed_down) for child in children.get(pid, ()))
    return found


@dataclass(frozen=True)
class TreeMemory:
    """A whole process tree's footprint: the root, its descendants, and
    the totals that do not double count."""

    root: Optional[ProcessMemory]
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


def read_tree_memory(root_pid: Optional[int] = None) -> TreeMemory:
    """Measure this process and everything below it.

    Cost is a page-table walk per process, so this belongs on a slow
    cadence (the run's periodic report), never in a hot path."""
    pid = os.getpid() if root_pid is None else root_pid
    roles = _classify(pid, _children_map(), dict(_game_processes))
    measured: list[ProcessMemory] = []
    for child in sorted(roles):
        process = read_process_memory(child)
        if process is not None:
            role, name = roles[child]
            measured.append(dataclasses.replace(process, role=role, name=name))
    root = read_process_memory(pid)
    return TreeMemory(
        root=None if root is None else dataclasses.replace(root, role=TAAF, name="main"),
        children=tuple(measured),
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

    The three components come from ``memory.stat`` and are what make
    ``current_kb`` interpretable. Page cache is charged to whoever first
    touched the file and is mapped by no process, so it is invisible to
    PSS and is most of any gap between the tree total and ``current_kb`` —
    loading tens of gigabytes of model weights puts them here."""

    path: str
    current_kb: int
    peak_kb: Optional[int]
    limit_kb: Optional[int]
    anon_kb: Optional[int] = None
    file_kb: Optional[int] = None
    kernel_kb: Optional[int] = None
    #: tmpfs pages. Charged inside ``file_kb`` by the kernel, but they have
    #: no backing file to be evicted to, so reclaiming them means swapping
    #: them out -- and a container with no swap cannot.
    shmem_kb: Optional[int] = None
    #: The cgroup's swap ceiling. ``None`` covers both "unreadable" and
    #: "unlimited", which are indistinguishable here because ``max`` parses
    #: to None -- and both are read as "assume no swap", the pessimistic
    #: side: this figure predicts a kill, so overstating it costs a second
    #: look and understating it costs the job.
    swap_max_kb: Optional[int] = None
    #: How many processes the kernel's OOM killer has killed in this cgroup
    #: (``memory.events``' ``oom_kill``), and how often usage reached the
    #: cap and forced reclaim (``max``). A process that died silently with a
    #: kill counted here was killed for memory; one without was not.
    oom_kills: Optional[int] = None
    limit_hits: Optional[int] = None

    @property
    def unreclaimable_kb(self) -> Optional[int]:
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

    cgroup: Optional[CgroupMemory]
    slurm_job_id: Optional[str]
    #: Cap SLURM was ASKED for, from the environment — the fallback when the
    #: cgroup files are not readable, and never preferred over them: the
    #: request is what we wanted, the cgroup is what we got.
    requested_kb: Optional[int]


def _read_int(path: Path) -> Optional[int]:
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


def _read_events(path: Path) -> dict[str, int]:
    """``memory.events`` (or v1's ``memory.oom_control``) as ``key -> count``.
    Counts, not sizes, so unlike ``_read_stat`` nothing is converted."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return {}
    events: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        try:
            events[key] = int(value)
        except ValueError:
            continue
    return events


def _own_cgroup_dir(proc_cgroup: Path, mount_root: Path) -> Optional[Path]:
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
) -> Optional[Path]:
    """This run's cgroup directory -- the SLURM job's where there is one,
    else this process's own -- or None where there is none to read."""
    own = _own_cgroup_dir(proc_cgroup, mount_root)
    return None if own is None else _job_cgroup_dir(own)


def read_cgroup_memory(
    proc_cgroup: Path = Path("/proc/self/cgroup"), mount_root: Path = Path("/sys/fs/cgroup")
) -> Optional[CgroupMemory]:
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
        events = _read_events(directory / "memory.events")
        oom_kills, limit_hits = events.get("oom_kill"), events.get("max")
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
        oom_kills, limit_hits = _read_events(directory / "memory.oom_control").get("oom_kill"), None
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
        oom_kills=oom_kills,
        limit_hits=limit_hits,
    )


def _requested_slurm_kb(env: Mapping[str, str]) -> Optional[int]:
    """The cap this job asked for, in kB, from SLURM's own environment."""

    def _int(name: str) -> Optional[int]:
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


def read_enforced_limit(environ: Optional[Mapping[str, str]] = None, **cgroup_kwargs: Path) -> Optional[EnforcedLimit]:
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


#: How many of the small processes the report names individually.
_TOP_PROCESSES = 10


def _describe(process: ProcessMemory) -> str:
    """``name role size`` with the process's peak and anonymous memory: a
    process at its peak is still growing or holding a high-water mark, and
    anonymous memory is the part a leak or a big payload lands in."""
    text = f"{process.name} {process.role} {_size(process.countable_kb)}"
    extra: list[str] = []
    if process.peak_kb is not None:
        extra.append(f"peak {_size(process.peak_kb)}")
    if process.anon_kb is not None:
        extra.append(f"anon {_size(process.anon_kb)}")
    return f"{text} ({', '.join(extra)})" if extra else text


def _largest(processes: list[ProcessMemory], count: int) -> list[ProcessMemory]:
    """The ``count`` largest by the countable figure, largest first."""
    return sorted(processes, key=lambda p: -p.countable_kb)[:count]


def format_breakdown_line(cgroup: CgroupMemory) -> Optional[str]:
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
    if cgroup.oom_kills is not None:
        parts.append(f"| oom kills {cgroup.oom_kills}")
        if cgroup.limit_hits is not None:
            parts.append(f"(cap reached {cgroup.limit_hits} times)")
    return " ".join(parts)


def format_memory_lines(tree: TreeMemory, limit: Optional[EnforcedLimit] = None) -> list[str]:
    """The report: the tree total split by role with the shared part called
    out, then TAAF and vLLM by name, then the spread per role of the small
    processes and the largest of them by name.

    The shared figure is on the line deliberately. Anyone who has looked
    at these processes individually has seen ~5 GB each and will read a
    much smaller total as wrong; naming how much is shared is what makes
    the total believable."""
    processes = tree.all_processes
    if not processes:
        return []
    total = tree.total_kb
    metric = "pss" if tree.has_pss else "rss (no smaps_rollup — shared pages ARE double counted)"
    by_role: dict[str, list[ProcessMemory]] = {}
    for process in processes:
        by_role.setdefault(process.role, []).append(process)
    # Split the total by role: PSS is additive, so the parts reconcile with
    # the tree exactly, and the one that is growing is named.
    parts: list[str] = []
    for role in (TAAF, VLLM, GAME, REPL, OTHER):
        members = by_role.get(role)
        if members:
            count = "" if role in (TAAF, VLLM) else f"{len(members)} "
            plural = "s" if count and len(members) > 1 and role != OTHER else ""
            parts.append(f"{count}{role}{plural} {_size(sum(p.countable_kb for p in members))}")
    line = f"tree {_size(total)} {metric}, {len(processes)} procs = " + " + ".join(parts)
    if tree.has_pss:
        shared = tree.rss_sum_kb - total
        line += f" (rss sums to {_size(tree.rss_sum_kb)}; {_size(shared)} of that is shared)"
    lines = [line]

    # The two big single owners, by name. Private is what dying would free.
    # Most of vLLM's footprint is its host KV tier in tmpfs: private while
    # one process maps it, shared once workers map it too.
    named: list[str] = []
    for role in (TAAF, VLLM):
        members = by_role.get(role)
        if members:
            # countable_kb falls back to RSS per process where PSS is
            # unreadable; name what the sum is made of.
            with_pss = sum(p.pss_kb is not None for p in members)
            unit = "pss" if with_pss == len(members) else "rss" if with_pss == 0 else "pss+rss"
            text = f"{role}: {_size(sum(p.countable_kb for p in members))} {unit}"
            if role == VLLM:
                text += " in " + ", ".join(f"{p.name} {_size(p.countable_kb)}" for p in _largest(members, len(members)))
            private = [p.uss_kb for p in members if p.uss_kb is not None]
            if len(private) == len(members):
                text += f", {_size(sum(private))} private"
            peaks = [p.peak_kb for p in members if p.peak_kb is not None]
            if role == TAAF and peaks:
                text += f", peak {_size(max(peaks))}"
            named.append(text)
    if named:
        lines.append(" | ".join(named))

    # Everything else is many small processes, one of which may not be:
    # the spread per role, then the largest by name.
    rest = [p for p in tree.children if p.role not in (TAAF, VLLM)]
    if rest:
        spreads: list[str] = []
        for role in (GAME, REPL, OTHER):
            members = by_role.get(role)
            if members:
                values = sorted(p.countable_kb for p in members)
                spreads.append(
                    f"{role}: mean {_size(sum(values) / len(values))} med {_size(_percentile(values, 0.5))} "
                    f"p90 {_size(_percentile(values, 0.9))} max {_size(values[-1])}"
                )
        lines.append(" | ".join(spreads))
        lines.append("top: " + ", ".join(_describe(p) for p in _largest(rest, _TOP_PROCESSES)))
    if limit is not None:
        lines.append(format_limit_line(limit, total))
        if limit.cgroup is not None:
            breakdown = format_breakdown_line(limit.cgroup)
            if breakdown is not None:
                lines.append(breakdown)
    return lines
