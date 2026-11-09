"""Full-weight draws carry immutable publication identity without a LoRA request."""

from dataclasses import replace

import pytest

import test_rollout_gateway as gateway_fixtures
from training.inference import _behavior_policy_snapshot, _validated_adapter_identity
from training.rollout_gateway import GatewayError, _adapter_identity


gateway = gateway_fixtures.gateway


def identity(**changes):
    return {
        "name": "stub-model",
        "id": 0,
        "path": None,
        "server_instance_id": "server-a",
        "kind": "full_weight",
        "weight_version": "a" * 32,
        "manifest_sha256": "b" * 64,
        **changes,
    }


def test_full_policy_group_identity_matches_across_servers_and_changes_with_weights():
    first = identity()
    second = identity(server_instance_id="server-b")

    def snapshot(value):
        return _behavior_policy_snapshot(value, model="base", revision="rev")

    assert snapshot(first) == snapshot(second)
    assert snapshot(first) != snapshot(identity(weight_version="c" * 32))
    assert _validated_adapter_identity(first, "stub-model") == first
    assert _adapter_identity(first, "stub-model") == first


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "lora"},
        {"weight_version": ""},
        {"weight_version": "a" * 31},
        {"manifest_sha256": "B" * 64},
        {"id": 1, "path": "/some/lora"},
        {"path": "/some/weights"},
        {"unknown": 1},
    ],
)
def test_invalid_publication_identity_is_rejected(changes):
    with pytest.raises((ValueError, RuntimeError)):
        _validated_adapter_identity(identity(**changes), "stub-model")
    with pytest.raises(GatewayError):
        _adapter_identity(identity(**changes), "stub-model")


@pytest.mark.parametrize("expected", [None, gateway_fixtures._adapter()])
def test_full_weight_admission_requires_a_completed_publication(gateway, expected):
    with pytest.raises(GatewayError, match="published full-weight"):
        gateway.set_policy(
            replace(
                gateway._policy, require_full_weight=True, expected_adapter=expected
            )
        )
    assert not gateway_fixtures._StubVllm.seen


@pytest.mark.parametrize("changed", [False, True])
def test_full_weight_draw_is_captured_and_stale_publication_is_fatal(gateway, changed):
    expected = identity()
    gateway.set_policy(
        replace(gateway._policy, require_full_weight=True, expected_adapter=expected)
    )
    observed = identity(weight_version="c" * 32) if changed else expected
    gateway_fixtures._StubVllm.response_changes = {"adapter": observed}
    response = gateway_fixtures._start(gateway, user="game#full-weight")
    if changed:
        assert response.status_code == 400
        with pytest.raises(GatewayError, match="identity"):
            gateway.drain_window()
    else:
        assert response.status_code == 200, response.text
        captures = gateway.drain_window().captures
        assert len(captures) == 1
        assert next(iter(captures.values())).adapter == expected


def policy_identity(**changes):
    return {
        "name": "stub-model",
        "id": 0,
        "path": None,
        "server_instance_id": "server-a",
        "kind": "policy_weights",
        "policy_sha256": "c" * 64,
        "policy_iteration": 4,
        **changes,
    }


def test_merged_policy_identity_names_the_publication_across_servers():
    first = policy_identity()

    def snapshot(value):
        return _behavior_policy_snapshot(value, model="base", revision="rev")

    assert snapshot(first) == snapshot(policy_identity(server_instance_id="server-b")) == {
        "model": "base", "revision": "rev", "kind": "policy_weights",
        "policy_sha256": "c" * 64, "policy_iteration": 4,
    }
    assert snapshot(first) != snapshot(policy_identity(policy_sha256="d" * 64))
    assert snapshot(first) != snapshot(identity())
    assert _validated_adapter_identity(first, "stub-model") == first
    assert _adapter_identity(first, "stub-model") == first
    assert _validated_adapter_identity(policy_identity(policy_iteration=-1), "stub-model")


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "full_weight"},
        {"policy_sha256": "C" * 64},
        {"policy_sha256": "c" * 63},
        {"policy_iteration": -2},
        {"policy_iteration": "4"},
        {"id": 1, "path": "/some/lora"},
        {"weight_version": "a" * 32},
    ],
)
def test_invalid_policy_identity_is_rejected(changes):
    with pytest.raises((ValueError, RuntimeError)):
        _validated_adapter_identity(policy_identity(**changes), "stub-model")
    with pytest.raises(GatewayError):
        _adapter_identity(policy_identity(**changes), "stub-model")


@pytest.mark.parametrize("expected", [None, gateway_fixtures._adapter(), identity()])
def test_flash_admission_requires_applied_policy_weights(gateway, expected):
    with pytest.raises(GatewayError, match="merged policy weights"):
        gateway.set_policy(
            replace(gateway._policy, require_lora=False, require_policy_weights=True, expected_adapter=expected)
        )
    assert not gateway_fixtures._StubVllm.seen


@pytest.mark.parametrize("changed", [False, True])
def test_merged_policy_draw_is_captured_and_another_publication_is_fatal(gateway, changed):
    expected = policy_identity()
    gateway.set_policy(
        replace(gateway._policy, require_lora=False, require_policy_weights=True, expected_adapter=expected)
    )
    observed = policy_identity(policy_sha256="d" * 64) if changed else expected
    gateway_fixtures._StubVllm.response_changes = {"adapter": observed}
    response = gateway_fixtures._start(gateway, user="game#merged")
    if changed:
        assert response.status_code == 400
        with pytest.raises(GatewayError, match="identity"):
            gateway.drain_window()
    else:
        assert response.status_code == 200, response.text
        captures = gateway.drain_window().captures
        assert next(iter(captures.values())).adapter == expected
