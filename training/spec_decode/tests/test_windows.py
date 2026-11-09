"""Long chains as windows: every reply position trains once, from the cache's rows."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from drafter.train import ChainData, _read, scored_from, windows


def _chain(tokens: int, generated: list[tuple[int, int]], path: Path = Path("unused")) -> ChainData:
    span = torch.full((tokens,), -1, dtype=torch.long)
    for call, (start, end) in enumerate(generated):
        span[start:end] = call
    return ChainData(
        chain_id="trace.0",
        harness="arc3",
        tokens=tokens,
        trained=sum(max(0, end - start - 1) for start, end in generated),
        path=path,
        logprobs_path=None,
        ids=torch.arange(tokens, dtype=torch.long) % 997,
        span=span,
        starts=torch.tensor([start for start, _ in generated], dtype=torch.long),
    )


def test_a_chain_that_fits_runs_whole():
    chain = _chain(3000, [(100, 2900)])
    assert windows(chain, 0) == [chain]
    assert windows(chain, 4096) == [chain]


def test_every_reply_position_trains_in_exactly_one_window():
    generated = [(100, 2000), (2500, 6000), (6200, 9990)]
    chain = _chain(10_000, generated)
    pieces = windows(chain, 4096)
    assert len(pieces) > 1
    trained_at = torch.zeros(chain.tokens, dtype=torch.long)
    for index, piece in enumerate(pieces):
        assert piece.tokens <= 4096 and piece.offset % 1024 == 0
        assert torch.equal(piece.ids, chain.ids[piece.offset : piece.offset + piece.tokens])
        where = (piece.span >= 0).nonzero().flatten()
        if index:
            # A later window's first quarter is context the head attends over.
            assert int(where.min()) >= 1024
        trained_at[where + piece.offset] += 1
        # Each call's reply in the window is one run from its start there.
        lengths = torch.bincount(piece.span[piece.span >= 0], minlength=len(generated))
        for call, n in enumerate(lengths.tolist()):
            if n:
                assert (piece.span[piece.starts[call] : piece.starts[call] + n] == call).all()
        assert piece.trained == int((lengths - 1).clamp(min=0).sum())
        assert piece.cached == chain.tokens
    assert torch.equal(trained_at, (chain.span >= 0).long())


def test_whole_chains_and_their_windows_score_the_same_positions_past_a_cut():
    generated = [(100, 2000), (2500, 6000), (6200, 9990)]
    chain = _chain(10_000, generated)
    cut = 4096

    def scored(pieces: list[ChainData]) -> torch.Tensor:
        at = torch.zeros(chain.tokens, dtype=torch.long)
        for piece in pieces:
            where = (piece.span >= 0).nonzero().flatten()
            at[where + piece.offset] += 1
            lengths = torch.bincount(piece.span[piece.span >= 0], minlength=len(generated))
            assert piece.trained == int((lengths - 1).clamp(min=0).sum()) > 0
            # A reply the cut falls in starts, for scoring, at the cut.
            for call, n in enumerate(lengths.tolist()):
                if n:
                    assert (piece.span[piece.starts[call] : piece.starts[call] + n] == call).all()
        return at

    whole = [piece for piece in [scored_from(chain, cut)] if piece is not None]
    pieces = [piece for window in windows(chain, 4096) if (piece := scored_from(window, cut)) is not None]
    expected = (chain.span >= 0).long()
    expected[:cut] = 0
    assert torch.equal(scored(whole), expected)
    assert torch.equal(scored(pieces), expected)
    # The whole chain keeps everything before the cut to attend over; the first window, which ends at it, has nothing to score.
    assert whole[0].tokens == chain.tokens and whole[0].offset == 0
    assert len(pieces) == len(windows(chain, 4096)) - 1
    assert scored_from(chain, 0) is chain
    assert scored_from(chain, chain.tokens) is None
    assert scored_from(_chain(5000, [(10, 3000)]), 4000) is None


def test_a_window_too_small_for_its_context_is_refused():
    with pytest.raises(ValueError):
        windows(_chain(5000, [(10, 4990)]), 2048)


def test_a_window_reads_its_rows_of_the_cached_stream(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        # _read pins what it reads for the copy to the card, which a CPU-only torch refuses.
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)

    tokens, width = 9000, 8
    path = tmp_path / "trace.0.safetensors"
    hidden = torch.randn(tokens, width).to(torch.bfloat16)
    chain = _chain(tokens, [(50, 8990)], path)
    save_file({"ids": chain.ids.to(torch.int32), "hidden": hidden}, str(path))
    _, whole, _ = _read(chain)
    assert torch.equal(whole, hidden)
    for piece in windows(chain, 4096):
        _, stream, _ = _read(piece)
        assert torch.equal(stream, hidden[piece.offset : piece.offset + piece.tokens])
    shorter = _chain(tokens - 1, [(50, 8000)], path)
    with pytest.raises(ValueError):
        _read(shorter)
