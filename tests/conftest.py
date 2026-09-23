"""Pytest configuration and test isolation fixtures for tool-cvat test suite.

Enforces:
1. Strict hermetic offline test execution:
   Guards against any unit test attempting socket connections to localhost:20128 (9Router),
   localhost:18080 (CVAT), or localhost:8070 (Nuclio dashboard) unless explicitly marked
   with '@pytest.mark.live'.
"""

from __future__ import annotations

import socket
from typing import Any
import pytest

FORBIDDEN_PORTS = {20128, 18080, 8070}
FORBIDDEN_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "host.docker.internal",
    "testserver",
}


def _is_forbidden_target(host: Any, port: int) -> bool:
    """Return True if (host, port) matches forbidden offline test targets."""
    if port not in FORBIDDEN_PORTS:
        return False
    if not isinstance(host, str):
        return True
    host_clean = host.lower().strip()
    return (
        host_clean in FORBIDDEN_HOSTS
        or host_clean.startswith("127.")
        or host_clean.endswith(".local")
        or host_clean == ""
    )


@pytest.fixture(autouse=True)
def hermetic_network_guard(request: pytest.FixtureRequest):
    """Network isolation fixture ensuring strict hermetic offline execution.

    Asserts that no unit test attempts socket connections to localhost:20128,
    localhost:18080, or localhost:8070 unless decorated with @pytest.mark.live.
    """
    if request.node.get_closest_marker("live"):
        yield
        return

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address: Any) -> Any:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if _is_forbidden_target(host, port):
                pytest.fail(
                    f"Hermetic offline test violation: Attempted socket connection to "
                    f"{host}:{port} without '@pytest.mark.live' mark on test '{request.node.nodeid}'. "
                    f"All calls to 9Router (:20128), CVAT (:18080), and Nuclio (:8070) must be mocked in unit tests."
                )
        return orig_connect(self, address)

    def guarded_connect_ex(self, address: Any) -> Any:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if _is_forbidden_target(host, port):
                pytest.fail(
                    f"Hermetic offline test violation: Attempted socket connection to "
                    f"{host}:{port} without '@pytest.mark.live' mark on test '{request.node.nodeid}'. "
                    f"All calls to 9Router (:20128), CVAT (:18080), and Nuclio (:8070) must be mocked in unit tests."
                )
        return orig_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
