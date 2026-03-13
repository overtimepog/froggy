"""Tests for streaming output filtering — thinking blocks and stop strings."""


from froggy.session import _STOP_STRINGS, strip_thinking


class TestStripThinking:
    def test_removes_complete_think_block(self):
        text = "<think>I need to figure this out.</think>Hello!"
        assert strip_thinking(text) == "Hello!"

    def test_removes_multiple_think_blocks(self):
        text = "<think>first</think>Hello <think>second</think>world"
        assert strip_thinking(text) == "Hello world"

    def test_removes_multiline_think_block(self):
        text = "<think>\nStep 1: analyze\nStep 2: respond\n</think>\nHere is my answer."
        result = strip_thinking(text)
        assert "<think>" not in result
        assert "Step 1" not in result
        assert "Here is my answer." in result

    def test_removes_unclosed_think_block(self):
        text = "Hello<think>still thinking about this..."
        assert strip_thinking(text) == "Hello"

    def test_no_think_blocks(self):
        text = "Just a normal response."
        assert strip_thinking(text) == "Just a normal response."

    def test_empty_string(self):
        assert strip_thinking("") == ""

    def test_only_think_block(self):
        text = "<think>just reasoning</think>"
        assert strip_thinking(text).strip() == ""

    def test_whitespace_after_think(self):
        text = "<think>reasoning</think>   Answer here."
        result = strip_thinking(text)
        assert "Answer here." in result

    def test_nested_angle_brackets_in_think(self):
        text = "<think>the user said <something></think>Response"
        assert strip_thinking(text) == "Response"


class TestStopStrings:
    def test_stop_strings_are_defined(self):
        assert len(_STOP_STRINGS) > 0

    def test_im_end_in_stop_strings(self):
        assert "<|im_end|>" in _STOP_STRINGS

    def test_im_start_in_stop_strings(self):
        assert "<|im_start|>" in _STOP_STRINGS

    def test_stop_string_detection(self):
        """Simulate what chat() does: detect stop strings and truncate."""
        raw = "Hello! How can I help you?<|im_end|><|im_start|>user"
        for stop in _STOP_STRINGS:
            if stop in raw:
                raw = raw[:raw.index(stop)]
                break
        assert raw == "Hello! How can I help you?"

    def test_stop_string_at_start(self):
        raw = "<|im_end|>extra stuff"
        for stop in _STOP_STRINGS:
            if stop in raw:
                raw = raw[:raw.index(stop)]
                break
        assert raw == ""

    def test_no_stop_string(self):
        raw = "Normal response without stop tokens."
        original = raw
        for stop in _STOP_STRINGS:
            if stop in raw:
                raw = raw[:raw.index(stop)]
                break
        assert raw == original
