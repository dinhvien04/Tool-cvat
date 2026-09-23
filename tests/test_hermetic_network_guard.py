"""Tests verifying strict hermetic network isolation in the test runner.

Ensures that:
1. Any socket connection attempt to localhost:20128, localhost:18080, or localhost:8070
   fails with a hermetic test violation unless decorated with @pytest.mark.live.
2. Tests decorated with @pytest.mark.live bypass the hermetic guard.
3. Other ports and non-forbidden addresses are not blocked by the guard.
"""

from __future__ import annotations

import socket
import pytest


def test_hermetic_guard_blocks_9router_port_unmarked():
    """Unmarked unit tests cannot open socket connections to localhost:20128."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        sock.connect(("127.0.0.1", 20128))
    assert "Hermetic offline test violation" in str(exc_info.value)
    assert "20128" in str(exc_info.value)


def test_hermetic_guard_blocks_cvat_port_unmarked():
    """Unmarked unit tests cannot open socket connections to localhost:18080."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        sock.connect(("localhost", 18080))
    assert "Hermetic offline test violation" in str(exc_info.value)
    assert "18080" in str(exc_info.value)


def test_hermetic_guard_blocks_nuclio_port_unmarked():
    """Unmarked unit tests cannot open socket connections to localhost:8070."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        sock.connect(("127.0.0.1", 8070))
    assert "Hermetic offline test violation" in str(exc_info.value)
    assert "8070" in str(exc_info.value)


def test_hermetic_guard_blocks_connect_ex():
    """Guards connect_ex on forbidden ports."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        sock.connect_ex(("127.0.0.1", 20128))
    assert "Hermetic offline test violation" in str(exc_info.value)


def test_hermetic_guard_blocks_remote_ip_on_forbidden_port():
    """Guards against any host IP attempting connections to forbidden internal ports."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        sock.connect(("192.168.1.100", 20128))
    assert "Hermetic offline test violation" in str(exc_info.value)
    assert "20128" in str(exc_info.value)


@pytest.mark.live
def test_hermetic_guard_allows_live_marked_test():
    """Tests decorated with @pytest.mark.live bypass the hermetic guard."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Underlying connect is attempted without pytest.fail; raises ConnectionRefused or Timeout
    try:
        sock.connect(("127.0.0.1", 20128))
    except (ConnectionRefusedError, OSError):
        pass  # Expected when 9Router is not running; crucial point is pytest.fail was NOT triggered
    finally:
        sock.close()
