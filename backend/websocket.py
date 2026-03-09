"""WebSocket connection manager for real-time communication."""

import logging
from collections import deque
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Max events buffered per session while WS is disconnected.
MAX_EVENT_BUFFER = 500


class ConnectionManager:
    """Manages WebSocket connections for multiple sessions."""

    def __init__(self) -> None:
        # session_id -> WebSocket
        self.active_connections: dict[str, WebSocket] = {}
        # session_id -> events buffered while WS was disconnected
        self._event_buffers: dict[str, deque[dict[str, Any]]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Accept a WebSocket connection, register it, and flush buffered events.

        Returns True if buffered events were flushed (i.e. this is a reconnect
        that had missed events).
        """
        logger.info(f"Attempting to accept WebSocket for session {session_id}")
        await websocket.accept()
        self.active_connections[session_id] = websocket

        # Flush events that were buffered while the WS was disconnected
        buffered = self._event_buffers.pop(session_id, None)
        if buffered:
            logger.info(
                f"Flushing {len(buffered)} buffered events for session {session_id}"
            )
            for message in buffered:
                try:
                    await websocket.send_json(message)
                except Exception:
                    logger.error(
                        f"Error flushing buffered event for session {session_id}"
                    )
                    break

        logger.info(f"WebSocket connected and registered for session {session_id}")
        return bool(buffered)

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    def clear_buffer(self, session_id: str) -> None:
        """Clear the event buffer for a session (e.g. on session delete)."""
        self._event_buffers.pop(session_id, None)

    async def send_event(
        self, session_id: str, event_type: str, data: dict[str, Any] | None = None
    ) -> None:
        """Send an event to a specific session's WebSocket.

        If no WebSocket is connected, the event is buffered so it can be
        replayed when the client reconnects.
        """
        message: dict[str, Any] = {"event_type": event_type}
        if data is not None:
            message["data"] = data

        if session_id not in self.active_connections:
            buf = self._event_buffers.setdefault(
                session_id, deque(maxlen=MAX_EVENT_BUFFER)
            )
            buf.append(message)
            return

        try:
            await self.active_connections[session_id].send_json(message)
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")
            self.disconnect(session_id)
            # Buffer the event that failed to send
            buf = self._event_buffers.setdefault(
                session_id, deque(maxlen=MAX_EVENT_BUFFER)
            )
            buf.append(message)

    async def broadcast(
        self, event_type: str, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast an event to all connected sessions."""
        for session_id in list(self.active_connections.keys()):
            await self.send_event(session_id, event_type, data)

    def is_connected(self, session_id: str) -> bool:
        """Check if a session has an active WebSocket connection."""
        return session_id in self.active_connections


# Global connection manager instance
manager = ConnectionManager()
