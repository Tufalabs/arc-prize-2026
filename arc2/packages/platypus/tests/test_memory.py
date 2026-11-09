"""Process-tree memory: the tree total must not double count pages that
several processes share, which is the whole reason RSS is the wrong
number here."""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from infra_fixtures import hold_mapped_file

from platypus import memory

_HAS_ROLLUP = Path("/proc/self/smaps_rollup").exists()
_needs_rollup = pytest.mark.skipif(not _HAS_ROLLUP, reason="no /proc/<pid>/smaps_rollup on this kernel")

MAPPED_BYTES = 64 * 1024 * 1024
MAPPED_KB = MAPPED_BYTES // 1024
SHARERS = 4


class _Sharers:
    """``SHARERS`` spawned children all holding one file mapped and touched.

    Measured differentially against an arm mapping a trivial file: each
    child is a full interpreter with platypus imported (~200 MB resident),
    which would otherwise swamp the 64 MiB under test. Differencing two
    arms that differ ONLY in the mapping cancels it exactly."""

    def __init__(self, path: Path, size: int):
        path.write_bytes(b"\xa5" * size)
        context = multiprocessing.get_context("spawn")
        self._stop = context.Event()
        readies = [context.Event() for _ in range(SHARERS)]
        self.processes = [
            context.Process(target=hold_mapped_file, args=(str(path), ready, self._stop)) for ready in readies
        ]
        for process in self.processes:
            process.start()
        for ready in readies:
            assert ready.wait(120), "a sharer never finished touching its mapping"

    def measure(self) -> memory.TreeMemory:
        pids = [process.pid for process in self.processes]
        assert all(pid is not None for pid in pids), "a sharer was never started"
        measured = [memory.read_process_memory(pid) for pid in pids if pid is not None]
        assert all(m is not None for m in measured), "a sharer died before it could be measured"
        return memory.TreeMemory(root=None, children=tuple(m for m in measured if m is not None))

    def close(self) -> None:
        self._stop.set()
        for process in self.processes:
            process.join(timeout=60)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


@pytest.fixture
def arms(tmp_path: Path) -> Iterator[tuple[memory.TreeMemory, memory.TreeMemory]]:
    """``(without the mapping, with it)`` — same processes, one variable."""
    control = _Sharers(tmp_path / "tiny.bin", 4096)
    try:
        subject = _Sharers(tmp_path / "big.bin", MAPPED_BYTES)
        try:
            yield control.measure(), subject.measure()
        finally:
            subject.close()
    finally:
        control.close()


@_needs_rollup
def test_pages_shared_between_processes_are_counted_once(
    arms: tuple[memory.TreeMemory, memory.TreeMemory],
) -> None:
    """The measurement this module exists for. Four processes each hold the
    same 64 MiB mapped: RSS says a quarter of a gigabyte became resident,
    the machine took 64 MiB, and only PSS agrees with the machine."""
    control, subject = arms
    assert subject.has_pss and control.has_pss

    rss_delta = subject.rss_sum_kb - control.rss_sum_kb
    pss_delta = subject.total_kb - control.total_kb

    # Naive RSS attributes the mapping to every sharer in full...
    assert rss_delta > SHARERS * MAPPED_KB * 0.8, f"rss only grew {rss_delta}kB across {SHARERS} sharers"
    # ...while the tree total grows by about one copy of it.
    assert MAPPED_KB * 0.5 < pss_delta < MAPPED_KB * 1.5, f"tree total grew {pss_delta}kB, want ~{MAPPED_KB}kB"
    # The report's own claim, in the report's own terms.
    assert (subject.rss_sum_kb - subject.total_kb) - (control.rss_sum_kb - control.total_kb) > 2.5 * MAPPED_KB


@_needs_rollup
def test_a_processs_private_total_excludes_what_it_shares(
    arms: tuple[memory.TreeMemory, memory.TreeMemory],
) -> None:
    """PSS answers "our share of this"; a process's own cost is USS, and
    the two must not be confused — a PSS figure falls when a SIBLING
    starts, with nothing freed. Shared file-backed pages belong in
    neither's private total."""
    control, subject = arms
    private_delta = sum(m.uss_kb or 0 for m in subject.children) - sum(m.uss_kb or 0 for m in control.children)
    assert private_delta < MAPPED_KB * 0.5, f"private totals grew {private_delta}kB for a shared mapping"


def test_the_tree_reaches_grandchildren(arms: tuple[memory.TreeMemory, memory.TreeMemory]) -> None:
    """A game child spawns kernel workers of its own, and the vLLM engine
    runs its core in a child of ours; a direct-children-only walk would
    miss the processes most likely to be holding the memory."""
    found = set(memory.descendants(os.getpid()))
    measured = {process_memory.pid for tree in arms for process_memory in tree.children}
    assert measured <= found


def test_the_report_names_how_much_is_shared() -> None:
    """Anyone who has looked at these processes one by one has seen ~5 GB
    each; a much smaller total reads as a bug unless the line says how much
    of the difference is sharing."""
    tree = memory.TreeMemory(
        root=memory.ProcessMemory(pid=1, rss_kb=5 * 1024 * 1024, pss_kb=1024 * 1024, uss_kb=900 * 1024),
        children=tuple(
            memory.ProcessMemory(pid=pid, rss_kb=5 * 1024 * 1024, pss_kb=1024 * 1024, uss_kb=512 * 1024)
            for pid in (2, 3, 4)
        ),
    )
    total, children = memory.format_memory_lines(tree)
    # The split has to reconcile: main + children == tree, exactly, because
    # PSS is additive. That is what makes "do the children explain the host
    # process's footprint?" answerable from one line.
    assert total.startswith("tree 4.0G pss, 4 procs = main 1.0G + children 3.0G")
    assert tree.root is not None
    assert tree.root.countable_kb + tree.children_total_kb == tree.total_kb
    assert "rss sums to 20.0G; 16.0G of that is shared" in total
    assert "3 children" in children and "private max 512M" in children


# --- the cap we are judged against ----------------------------------------


def _v2_tree(tmp_path: Path, cgroup_path: str, **files: str) -> tuple[Path, Path]:
    """A fake unified-hierarchy ``(proc_cgroup, mount_root)`` pair."""
    mount_root = tmp_path / "sys"
    directory = mount_root / cgroup_path.lstrip("/")
    directory.mkdir(parents=True)
    for name, value in files.items():
        (directory / name.replace("_", ".", 1)).write_text(value + "\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text(f"0::{cgroup_path}\n")
    return proc_cgroup, mount_root


def test_the_job_cgroup_is_read_not_this_steps(tmp_path: Path) -> None:
    """Our own cgroup is the STEP. A job running several steps would report
    a fraction of what it is judged on, which is the wrong number to put
    next to a cap."""
    proc_cgroup, mount_root = _v2_tree(
        tmp_path,
        "/system.slice/slurmstepd.scope/job_4242/step_0/user",
        memory_current=str(3 * 1024**3),
        memory_max="max",
    )
    job = mount_root / "system.slice/slurmstepd.scope/job_4242"
    (job / "memory.current").write_text(str(9 * 1024**3) + "\n")
    (job / "memory.max").write_text(str(16 * 1024**3) + "\n")
    (job / "memory.peak").write_text(str(11 * 1024**3) + "\n")

    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None
    assert cgroup.path.endswith("job_4242")
    assert cgroup.current_kb == 9 * 1024**2 and cgroup.limit_kb == 16 * 1024**2
    assert cgroup.peak_kb == 11 * 1024**2


def test_an_uncapped_cgroup_reports_no_cap_rather_than_a_huge_one(tmp_path: Path) -> None:
    # v2 spells it "max"; a naive int() would report 8 exabytes as the cap
    # and every run would look comfortably within it.
    proc_cgroup, mount_root = _v2_tree(
        tmp_path, "/user.slice/session.scope", memory_current=str(1024**3), memory_max="max"
    )
    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None and cgroup.limit_kb is None


def test_cgroup_v1_is_read_and_its_unlimited_sentinel_understood(tmp_path: Path) -> None:
    """v1 writes "no limit" as a page-counter maximum, not a word."""
    mount_root = tmp_path / "sys"
    directory = mount_root / "memory" / "slurm/uid_1000/job_77/step_0"
    directory.mkdir(parents=True)
    (directory / "memory.usage_in_bytes").write_text(str(2 * 1024**3) + "\n")
    (directory / "memory.max_usage_in_bytes").write_text(str(3 * 1024**3) + "\n")
    (directory / "memory.limit_in_bytes").write_text("9223372036854710784\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text("11:memory,cpuacct:/slurm/uid_1000/job_77/step_0\n1:name=systemd:/\n")

    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None
    assert cgroup.current_kb == 2 * 1024**2
    assert cgroup.limit_kb is None, "the v1 sentinel was reported as a real cap"


def test_no_cap_and_no_job_reports_nothing(tmp_path: Path) -> None:
    # On a dev box the line would be pure noise, so there must not be one.
    proc_cgroup, mount_root = _v2_tree(
        tmp_path, "/user.slice/session.scope", memory_current=str(1024**3), memory_max="max"
    )
    assert memory.read_enforced_limit({}, proc_cgroup=proc_cgroup, mount_root=mount_root) is None


def test_the_requested_cap_is_only_a_fallback(tmp_path: Path) -> None:
    """What SLURM was asked for and what the cgroup enforces can differ;
    the enforced one is what kills the run, so it must win."""
    proc_cgroup, mount_root = _v2_tree(
        tmp_path, "/job_9/step_0", memory_current=str(4 * 1024**3), memory_max=str(8 * 1024**3)
    )
    env = {"SLURM_JOB_ID": "9", "SLURM_MEM_PER_NODE": str(64 * 1024)}
    limit = memory.read_enforced_limit(env, proc_cgroup=proc_cgroup, mount_root=mount_root)
    assert limit is not None and limit.cgroup is not None
    line = memory.format_limit_line(limit, tree_total_kb=3 * 1024**2)
    assert "slurm job 9" in line
    assert "of 8.0G cap" in line and "64.0G" not in line

    # With no cgroup to read, the request is all there is — and the line
    # must say that rather than implying the number is enforced.
    blind = memory.read_enforced_limit(env, proc_cgroup=tmp_path / "missing", mount_root=mount_root)
    assert blind is not None and blind.cgroup is None
    assert "64.0G cap requested" in memory.format_limit_line(blind, tree_total_kb=3 * 1024**2)


def test_the_limit_line_says_how_much_of_the_cgroup_we_account_for() -> None:
    """The cgroup carries page cache and kernel memory we caused but hold in
    no process, so it sits above our tree total. Naming the gap is what
    stops an approaching OOM from reading as a contradiction."""
    limit = memory.EnforcedLimit(
        cgroup=memory.CgroupMemory(
            path="/job_1", current_kb=71 * 1024**2, peak_kb=78 * 1024**2, limit_kb=120 * 1024**2
        ),
        slurm_job_id="1",
        requested_kb=None,
    )
    line = memory.format_limit_line(limit, tree_total_kb=61 * 1024**2)
    assert "holding 71.0G of 120.0G cap (59%)" in line
    assert "peak 78.0G" in line
    assert "our tree accounts for 61.0G" in line


def test_the_breakdown_separates_what_reclaim_could_free(tmp_path: Path) -> None:
    """A real job: 46.1G held against a 64G cap reads as 72% and alarming,
    but a third of it is page cache from reading the model weights, which
    the kernel drops before killing anything. The percentage that predicts
    an OOM is the unreclaimable one, so the line has to carry it."""
    proc_cgroup, mount_root = _v2_tree(
        tmp_path,
        "/job_1633958/step_0",
        memory_current=str(46 * 1024**3),
        memory_max=str(64 * 1024**3),
    )
    (mount_root / "job_1633958/step_0/memory.stat").write_text(
        f"anon {31 * 1024**3}\nfile {13 * 1024**3}\nkernel {2 * 1024**3}\npagetables {1024**3}\n"
    )
    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None
    assert cgroup.anon_kb == 31 * 1024**2 and cgroup.file_kb == 13 * 1024**2
    assert cgroup.unreclaimable_kb == 33 * 1024**2

    line = memory.format_breakdown_line(cgroup)
    assert line is not None
    assert "anon 31.0G + page cache 13.0G + kernel 2.0G" in line
    # 33 of 64 — half the cap, against the 72% the total alone reports.
    assert "33.0G would survive reclaim (52% of cap)" in line


def test_cgroup_v1_field_names_are_translated(tmp_path: Path) -> None:
    """v1 calls them ``rss`` and ``cache``; reading v2's names there would
    silently drop the breakdown on any v1 cluster."""
    mount_root = tmp_path / "sys"
    directory = mount_root / "memory" / "slurm/uid_1000/job_5/step_0"
    directory.mkdir(parents=True)
    (directory / "memory.usage_in_bytes").write_text(str(10 * 1024**3))
    (directory / "memory.limit_in_bytes").write_text(str(20 * 1024**3))
    (directory / "memory.stat").write_text(f"cache {4 * 1024**3}\nrss {6 * 1024**3}\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text("9:memory:/slurm/uid_1000/job_5/step_0\n")

    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None
    assert cgroup.anon_kb == 6 * 1024**2 and cgroup.file_kb == 4 * 1024**2


def test_an_unreadable_breakdown_still_reports_the_total(tmp_path: Path) -> None:
    # The breakdown is an enrichment; losing it must not cost the cap line.
    proc_cgroup, mount_root = _v2_tree(
        tmp_path, "/job_7/step_0", memory_current=str(4 * 1024**3), memory_max=str(8 * 1024**3)
    )
    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None and cgroup.current_kb == 4 * 1024**2
    assert cgroup.unreclaimable_kb is None
    assert memory.format_breakdown_line(cgroup) is None


def test_a_kernel_without_pss_says_so_rather_than_reporting_a_wrong_total() -> None:
    # Falling back silently would report a number that IS double counted
    # while looking exactly like one that is not.
    tree = memory.TreeMemory(root=memory.ProcessMemory(pid=1, rss_kb=2048, pss_kb=None, uss_kb=None), children=())
    (line,) = memory.format_memory_lines(tree)
    assert "double counted" in line
    assert "of that is shared" not in line


def test_shared_memory_is_counted_as_unreclaimable(tmp_path: Path) -> None:
    """The kernel files tmpfs under ``file`` but cannot drop it without swap,
    so a host KV tier claimed in ``/dev/shm`` has to count with the anon
    memory: read as cache, a job holding sixty gigabytes of tier reports
    sixty gigabytes of headroom it does not have."""
    gib = 1024**3
    proc_cgroup, mount_root = _v2_tree(
        tmp_path,
        "/kubepods.slice/kernel.scope",
        memory_current=str(129 * gib),
        memory_max=str(173 * gib),
        memory_stat=f"anon {20 * gib}\nfile {107 * gib}\nkernel {1 * gib}\nshmem {60 * gib}",
    )
    cgroup = memory.read_cgroup_memory(proc_cgroup, mount_root)
    assert cgroup is not None
    assert cgroup.shmem_kb == 60 * 1024**2
    assert cgroup.unreclaimable_kb == (20 + 1 + 60) * 1024**2
    line = memory.format_breakdown_line(cgroup)
    assert line is not None
    assert f"tmpfs {memory._size(60 * 1024**2)}" in line
    assert f"{memory._size(81 * 1024**2)} would survive reclaim" in line


def test_tmpfs_is_not_reclaimable_without_swap() -> None:
    """The kernel charges tmpfs inside ``file``, and it is not page cache:
    a host KV tier or a /dev/shm buffer has no backing file to be evicted
    to, so freeing it means swapping it out. Counting it as cache is how a
    job reads as comfortable while two thirds of its cap is spoken for."""
    with_tmpfs = memory.CgroupMemory(
        path="/x", current_kb=100, peak_kb=100, limit_kb=1000, anon_kb=10, file_kb=80, kernel_kb=1, shmem_kb=60
    )
    assert with_tmpfs.unreclaimable_kb == 71, "anon + kernel + tmpfs"

    # With swap the kernel can page it out, so it is reclaimable after all.
    swappable = dataclasses.replace(with_tmpfs, swap_max_kb=4096)
    assert swappable.unreclaimable_kb == 11, "anon + kernel only"

    # And a cgroup that reports no tmpfs is unaffected either way.
    none = dataclasses.replace(with_tmpfs, shmem_kb=0)
    assert none.unreclaimable_kb == 11
