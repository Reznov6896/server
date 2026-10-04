"""Tests for waiting until a DLNA renderer is ready to play a new URI."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from async_upnp_client.profiles.dlna import TransportState
from music_assistant_models.enums import MediaType
from music_assistant_models.player import PlayerMedia

from music_assistant.providers.dlna import player as dlna_player
from music_assistant.providers.dlna.player import DLNAPlayer
from tests.common import MockProvider

if TYPE_CHECKING:
    import pytest


def _mock_device(**device_state: Any) -> MagicMock:
    """Return a mocked renderer that does not offer Play yet."""
    device = MagicMock()
    device.can_play = False
    device.transport_state = TransportState.STOPPED
    device._async_poll_state_variables = AsyncMock()
    for name, value in device_state.items():
        setattr(device, name, value)
    return device


def _player(device: MagicMock) -> DLNAPlayer:
    """Return a player for the given renderer."""
    provider = MockProvider("dlna", instance_id="dlna_test")
    provider.mass.streams.base_url = "http://192.168.1.2:8097"
    provider.mass.streams.resolve_stream_url = AsyncMock(
        return_value="http://192.168.1.2:8097/flow/dlna-player.flac"
    )
    return DLNAPlayer(
        provider,  # type: ignore[arg-type]
        "uuid:dlna-player",
        "http://192.168.1.10/description.xml",
        device=device,
    )


async def test_returns_at_once_when_play_is_offered() -> None:
    """A renderer that offers Play is not polled at all."""
    device = _mock_device(can_play=True)
    start = time.monotonic()
    await _player(device)._wait_for_can_play(10)
    assert time.monotonic() - start < 0.2
    device._async_poll_state_variables.assert_not_awaited()


async def test_returns_at_once_when_already_playing() -> None:
    """A renderer that already plays the new URI does not need a Play command."""
    device = _mock_device(transport_state=TransportState.PLAYING)
    start = time.monotonic()
    await _player(device)._wait_for_can_play(10)
    assert time.monotonic() - start < 0.2
    device._async_poll_state_variables.assert_not_awaited()


async def test_stops_waiting_when_renderer_keeps_transitioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renderer stuck in TRANSITIONING without offering Play no longer costs the full timeout."""
    monkeypatch.setattr(dlna_player, "TRANSITIONING_GRACE_TIME", 0.5)
    device = _mock_device(transport_state=TransportState.TRANSITIONING)
    start = time.monotonic()
    await _player(device)._wait_for_can_play(10)
    assert time.monotonic() - start < 2
    assert device._async_poll_state_variables.await_count >= 1


async def test_waits_until_play_is_offered() -> None:
    """A stopped renderer is polled until it offers Play."""
    device = _mock_device()

    async def _offer_play(*_args: Any, **_kwargs: Any) -> None:
        device.can_play = True

    device._async_poll_state_variables = AsyncMock(side_effect=_offer_play)
    await _player(device)._wait_for_can_play(10)
    assert device._async_poll_state_variables.await_count == 1


async def test_gives_up_after_max_wait_time() -> None:
    """A renderer that never gets ready is waited for no longer than the timeout."""
    device = _mock_device()
    start = time.monotonic()
    await _player(device)._wait_for_can_play(0.6)
    elapsed = time.monotonic() - start
    assert 0.5 <= elapsed < 1.5
    assert device._async_poll_state_variables.await_count >= 2


async def test_play_media_is_not_blocked_by_transitioning_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """play_media() does not wait for Play on a renderer that starts playing by itself."""
    monkeypatch.setattr(dlna_player, "TRANSITIONING_GRACE_TIME", 0.5)
    device = _mock_device(transport_state=TransportState.TRANSITIONING, can_stop=False)
    device.async_set_transport_uri = AsyncMock()
    device.async_play = AsyncMock()

    async def _never_offers_play(max_wait_time: float = 5) -> None:
        # what DmrDevice.async_wait_for_can_play() does on such a renderer: run into its timeout
        await asyncio.sleep(max_wait_time)

    device.async_wait_for_can_play = AsyncMock(side_effect=_never_offers_play)
    player = _player(device)
    start = time.monotonic()
    await player.play_media(
        PlayerMedia(uri="library://track/1", media_type=MediaType.TRACK, title="Track")
    )
    assert time.monotonic() - start < 2
    device.async_set_transport_uri.assert_awaited_once()
    device.async_play.assert_awaited_once()
