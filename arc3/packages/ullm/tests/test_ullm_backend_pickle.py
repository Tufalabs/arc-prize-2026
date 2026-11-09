"""Backend picklability (U1/U6: picklable specs -> per-process live clients).

Backends pickle as configuration: live clients are rebuilt on unpickle,
per-process accounting state restarts at zero, and unpicklable situations
(requests in flight, started factories) fail loudly instead of producing a
silently degraded copy.
"""

import copy
import pickle

import pytest
import ullm.backend
import ullm.backends.anthropic
import ullm.backends.oai
import ullm.backends.stub
import ullm.factory


def test_dummy_backend_config_roundtrip_and_accounting_reset():
    b = ullm.backend.DummyBackend(send_back_reasoning="all", track_warnings=True)
    b.log_warning({"type": "tool_error", "detail": "x"})
    b.total_usage += ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=7, num_requests=2)
    b._request_counter = 2

    b2 = pickle.loads(pickle.dumps(b))

    assert isinstance(b2, ullm.backend.DummyBackend)
    assert b2.send_back_reasoning == "all"  # config survives
    # Per-process accounting restarts at zero on the receiving side.
    assert b2.warning_log == []
    assert b2.total_usage.prompt_tokens == 0 and b2.total_usage.num_requests == 0
    assert b2._request_counter == 0
    # The copy is functional: warnings can be logged against fresh state.
    b2.log_warning({"type": "tool_error"})
    assert len(b2.warning_log) == 1 and b.warning_log[0]["detail"] == "x"


def test_pickle_with_request_in_flight_fails_loudly():
    b = ullm.backend.DummyBackend()
    b._in_flight = 1
    with pytest.raises(AssertionError, match="in flight"):
        pickle.dumps(b)


def test_deepcopy_detaches_as_config_clone():
    # Deepcopy = pickle semantics: a detached configuration clone with
    # fresh accounting, never a silent share (sharing must be explicit via
    # a seeded memo — see Backend.__getstate__).
    b = ullm.backend.DummyBackend(send_back_reasoning="all")
    b.total_usage += ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=7, num_requests=2)
    clone = copy.deepcopy({"backend": b})["backend"]
    assert clone is not b
    assert clone.send_back_reasoning == "all"  # config survives
    assert clone.total_usage.num_requests == 0  # accounting detached
    assert b.total_usage.num_requests == 2  # original untouched


def test_deepcopy_with_seeded_memo_shares_the_instance():
    b = ullm.backend.DummyBackend()
    holder = {"backend": b}
    assert copy.deepcopy(holder, {id(b): b})["backend"] is b


def test_openai_server_rebuilds_client():
    b = ullm.backends.oai.OpenAIServer(
        base_url="http://localhost:9999/v1", api_key="k", model="m", parse_think_tags=True
    )
    b2 = pickle.loads(pickle.dumps(b))
    assert b2.parse_think_tags is True and b2.model == "m"
    # The client is a fresh process-local object pointing at the same server.
    assert b2._client is not b._client
    assert str(b2._client.base_url).startswith("http://localhost:9999/v1")


def test_anthropic_rebuilds_client():
    b = ullm.backends.anthropic.Anthropic(api_key="test-key", model="claude-haiku-4-5", streaming=False)
    b2 = pickle.loads(pickle.dumps(b))
    assert b2.streaming is False and b2.model == "claude-haiku-4-5"
    assert b2._client is not b._client
    assert b2._client.api_key == "test-key"
    # __post_init__-derived config (pricing lookup) survives the round-trip.
    assert b2.pricing.input_per_mtok == b.pricing.input_per_mtok


def test_stub_backend_script_survives_pickle():
    b = ullm.backends.stub.StubBackend.from_script("one", "two")
    b2 = pickle.loads(pickle.dumps(b))
    assert [r.final_content.to_string() for r in b2.responses] == ["one", "two"]


async def test_prebuilt_backend_factory_lifecycle():
    factory = ullm.factory.PrebuiltBackend(backend=ullm.backend.DummyBackend())
    # Picklable as a spec before start.
    factory2 = pickle.loads(pickle.dumps(factory))
    assert not factory2.started
    backend = await factory2.start()
    assert factory2.started
    assert isinstance(backend, ullm.backend.DummyBackend)
    # A started factory holds live handles: pickling it must fail loudly,
    # and starting twice is a bug.
    with pytest.raises(AssertionError, match="started"):
        pickle.dumps(factory2)
    with pytest.raises(AssertionError, match="already started"):
        await factory2.start()
    await factory2.close()
    assert not factory2.started
    # close() resets: the factory can be pickled/shipped again.
    pickle.dumps(factory2)


async def test_factory_context_manager():
    async with ullm.factory.PrebuiltBackend(backend=ullm.backend.DummyBackend()) as backend:
        assert isinstance(backend, ullm.backend.DummyBackend)
