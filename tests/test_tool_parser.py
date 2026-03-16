"""Tests for ToolCallParser."""


from froggy.tool_parser import ToolCallParser, parse_tool_calls


class TestParseToolCalls:
    def test_hermes_xml_single_call(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/foo.txt"}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "/tmp/foo.txt"}

    def test_hermes_xml_no_args(self):
        text = '<tool_call>{"name": "get_datetime", "arguments": {}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_datetime"
        assert calls[0].arguments == {}

    def test_hermes_xml_multiple_calls(self):
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'
            " some text "
            '<tool_call>{"name": "write_file", "arguments": {"path": "b.txt", "content": "hi"}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "read_file"
        assert calls[1].name == "write_file"

    def test_bare_json_fallback(self):
        text = '{"name": "run_shell", "arguments": {"command": "ls -la"}}'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "run_shell"
        assert calls[0].arguments == {"command": "ls -la"}

    def test_bare_json_embedded_in_text(self):
        text = 'Sure! {"name": "get_datetime", "arguments": {"tz": "UTC"}} done.'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_datetime"

    def test_no_tool_call_returns_empty(self):
        text = "Hello, how can I help you today?"
        calls = parse_tool_calls(text)
        assert calls == []

    def test_empty_string(self):
        assert parse_tool_calls("") == []

    def test_invalid_json_in_xml_skipped(self):
        text = "<tool_call>not valid json</tool_call>"
        calls = parse_tool_calls(text)
        assert calls == []

    def test_hermes_xml_with_surrounding_text(self):
        text = (
            "I'll read that file for you.\n"
            '<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>\n'
            "Done."
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"

    def test_tool_call_raw_field_xml(self):
        raw_xml = '<tool_call>{"name": "get_datetime", "arguments": {}}</tool_call>'
        calls = parse_tool_calls(raw_xml)
        assert calls[0].raw == raw_xml

    def test_tool_call_raw_field_bare_json(self):
        raw = '{"name": "run_shell", "arguments": {"command": "pwd"}}'
        calls = parse_tool_calls(raw)
        assert calls[0].raw == raw

    def test_multiline_arguments(self):
        text = (
            "<tool_call>"
            '{"name": "write_file", "arguments": {"path": "out.txt", "content": "line1\\nline2"}}'
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments["content"] == "line1\nline2"

    def test_json_without_arguments_key_ignored(self):
        # Bare JSON without "arguments" key should not be picked up
        text = '{"name": "something", "other": "value"}'
        calls = parse_tool_calls(text)
        assert calls == []


class TestToolCallParserStreaming:
    def test_streaming_complete_in_one_chunk(self):
        parser = ToolCallParser()
        calls = parser.feed('<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>')
        assert len(calls) == 1
        assert calls[0].name == "read_file"

    def test_streaming_split_across_chunks(self):
        parser = ToolCallParser()
        chunks = [
            "<tool_",
            "call>",
            '{"name": "read_file",',
            ' "arguments": {"path": "x.py"}}',
            "</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            all_calls.extend(parser.feed(chunk))
        all_calls.extend(parser.flush())
        assert len(all_calls) == 1
        assert all_calls[0].name == "read_file"

    def test_streaming_two_calls_across_chunks(self):
        parser = ToolCallParser()
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'
            '<tool_call>{"name": "get_datetime", "arguments": {}}</tool_call>'
        )
        # Feed character by character
        all_calls = []
        for ch in text:
            all_calls.extend(parser.feed(ch))
        all_calls.extend(parser.flush())
        assert len(all_calls) == 2

    def test_reset_clears_buffer(self):
        parser = ToolCallParser()
        parser.feed("<tool_call>")
        assert parser.buffer != ""
        parser.reset()
        assert parser.buffer == ""

    def test_flush_empty_buffer_returns_empty(self):
        parser = ToolCallParser()
        assert parser.flush() == []

    def test_buffer_property(self):
        parser = ToolCallParser()
        parser.feed("hello")
        assert parser.buffer == "hello"

    def test_no_false_positives_in_streaming_prose(self):
        parser = ToolCallParser()
        prose = "The weather is nice today. I recommend going outside."
        calls = parser.feed(prose)
        calls.extend(parser.flush())
        assert calls == []
