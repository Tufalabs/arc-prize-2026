"""Native serving cannot expose an unauthenticated network listener."""

import pytest

from training.native_vllm_server import validate_bind_auth


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.4.5.6", "::1", "localhost", "LOCALHOST."]
)
def test_loopback_bind_can_omit_auth(host):
    validate_bind_auth(host, None)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "", "node-a", "192.168.1.1", "::ffff:192.168.1.1"]
)
@pytest.mark.parametrize("key", [None, "", "   "])
def test_network_bind_requires_nonempty_auth(host, key):
    with pytest.raises(RuntimeError, match="SERVER_API_KEY"):
        validate_bind_auth(host, key)


def test_authenticated_network_bind():
    validate_bind_auth("0.0.0.0", "test-key")


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_auth_whitespace_rejected_before_client_server_disagreement(host):
    with pytest.raises(RuntimeError, match="surrounding whitespace"):
        validate_bind_auth(host, " example ")
