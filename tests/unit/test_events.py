"""Tests for event emitter."""

from universal_ai.events import EventEmitter


class TestEventEmitter:
    def test_on_register_handler(self):
        emitter = EventEmitter()

        def handler(x):
            return None

        emitter.on("test", handler)
        assert emitter.handler_count("test") == 1

    def test_multiple_handlers(self):
        emitter = EventEmitter()
        emitter.on("test", lambda x: None)
        emitter.on("test", lambda x: None)
        assert emitter.handler_count("test") == 2

    def test_off_removes_handler(self):
        emitter = EventEmitter()

        def handler(x):
            return None

        emitter.on("test", handler)
        result = emitter.off("test", handler)
        assert result is True
        assert emitter.handler_count("test") == 0

    def test_off_returns_false_if_not_found(self):
        emitter = EventEmitter()

        def handler(x):
            return None

        result = emitter.off("test", handler)
        assert result is False

    def test_emit_calls_handlers(self):
        emitter = EventEmitter()
        results = []
        emitter.on("test", results.append)
        emitter.emit("test", "hello")
        assert results == ["hello"]

    def test_emit_multiple_handlers(self):
        emitter = EventEmitter()
        results = []
        emitter.on("test", results.append)
        emitter.on("test", lambda x: results.append(x * 2))
        emitter.emit("test", 5)
        assert results == [5, 10]

    def test_emit_different_events(self):
        emitter = EventEmitter()
        results = []
        emitter.on("a", lambda x: results.append("a"))
        emitter.on("b", lambda x: results.append("b"))
        emitter.emit("a", None)
        assert results == ["a"]

    def test_clear_all(self):
        emitter = EventEmitter()
        emitter.on("a", lambda x: None)
        emitter.on("b", lambda x: None)
        emitter.clear()
        assert len(emitter.events()) == 0

    def test_clear_specific_event(self):
        emitter = EventEmitter()
        emitter.on("a", lambda x: None)
        emitter.on("b", lambda x: None)
        emitter.clear("a")
        assert emitter.handler_count("a") == 0
        assert emitter.handler_count("b") == 1

    def test_events_list(self):
        emitter = EventEmitter()
        emitter.on("a", lambda x: None)
        emitter.on("b", lambda x: None)
        assert sorted(emitter.events()) == ["a", "b"]

    def test_repr(self):
        emitter = EventEmitter()
        assert "EventEmitter" in repr(emitter)

    def test_handler_exception_isolated(self):
        emitter = EventEmitter()
        results = []

        def bad_handler(x):
            msg = "error"
            raise ValueError(msg)

        def good_handler(x):
            results.append(x)

        emitter.on("test", bad_handler)
        emitter.on("test", good_handler)
        emitter.emit("test", "hello")
        assert results == ["hello"]
