"""Tests for ToolRegistry, ToolDef, and ToolSelector."""

import json

from froggy.tool_selector import ToolSelector
from froggy.tools import CORE_TOOLS, ToolDef, ToolParam, ToolRegistry


class TestToolDef:
    def test_to_json_schema_basic(self):
        tool = ToolDef(
            name="read_file",
            description="Read a file.",
            params=[ToolParam("path", "string", "File path.")],
        )
        schema = tool.to_json_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "read_file"
        assert fn["description"] == "Read a file."
        assert "path" in fn["parameters"]["properties"]
        assert "path" in fn["parameters"]["required"]

    def test_to_json_schema_optional_param(self):
        tool = ToolDef(
            name="list_directory",
            description="List a directory.",
            params=[ToolParam("path", "string", "Dir path.", required=False)],
        )
        schema = tool.to_json_schema()
        assert "path" not in schema["function"]["parameters"]["required"]
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_to_json_schema_enum(self):
        tool = ToolDef(
            name="set_mode",
            description="Set mode.",
            params=[ToolParam("mode", "string", "Mode.", enum=["fast", "slow"])],
        )
        schema = tool.to_json_schema()
        prop = schema["function"]["parameters"]["properties"]["mode"]
        assert prop["enum"] == ["fast", "slow"]

    def test_no_params(self):
        tool = ToolDef(name="get_datetime", description="Get time.", params=[])
        schema = tool.to_json_schema()
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_multiple_params(self):
        tool = ToolDef(
            name="write_file",
            description="Write a file.",
            params=[
                ToolParam("path", "string", "Path."),
                ToolParam("content", "string", "Content."),
            ],
        )
        schema = tool.to_json_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert "content" in props
        assert schema["function"]["parameters"]["required"] == ["path", "content"]


class TestToolRegistry:
    def test_default_contains_core_tools(self):
        registry = ToolRegistry()
        assert set(registry.names()) == {t.name for t in CORE_TOOLS}

    def test_default_has_six_tools(self):
        registry = ToolRegistry()
        assert len(registry.all()) == 6

    def test_register_and_get(self):
        registry = ToolRegistry(tools=[])
        tool = ToolDef("mytool", "Does something.", [])
        registry.register(tool)
        assert registry.get("mytool") is tool

    def test_get_missing_returns_none(self):
        registry = ToolRegistry(tools=[])
        assert registry.get("nonexistent") is None

    def test_custom_tools(self):
        tools = [ToolDef("a", "Tool A.", []), ToolDef("b", "Tool B.", [])]
        registry = ToolRegistry(tools=tools)
        assert registry.names() == ["a", "b"]

    def test_to_json_schema_returns_list(self):
        registry = ToolRegistry()
        schema = registry.to_json_schema()
        assert isinstance(schema, list)
        assert len(schema) == len(CORE_TOOLS)
        for item in schema:
            assert item["type"] == "function"

    def test_to_json_schema_empty_registry(self):
        registry = ToolRegistry(tools=[])
        assert registry.to_json_schema() == []

    def test_to_gbnf_grammar_contains_tool_names(self):
        registry = ToolRegistry()
        grammar = registry.to_gbnf_grammar()
        for name in registry.names():
            assert name in grammar
        assert "root" in grammar
        assert "tool-name-val" in grammar

    def test_to_gbnf_grammar_has_required_rules(self):
        registry = ToolRegistry()
        grammar = registry.to_gbnf_grammar()
        assert "json-obj" in grammar
        assert "json-string" in grammar
        assert "ws" in grammar

    def test_system_prompt_block_contains_tool_names(self):
        registry = ToolRegistry()
        prompt = registry.system_prompt_block()
        for name in registry.names():
            assert name in prompt
        assert "<tool_call>" in prompt

    def test_system_prompt_block_valid_json(self):
        registry = ToolRegistry()
        prompt = registry.system_prompt_block()
        json_start = prompt.index("[")
        json_str = prompt[json_start:]
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == len(CORE_TOOLS)

    def test_all_returns_all_tools(self):
        registry = ToolRegistry()
        assert len(registry.all()) == len(CORE_TOOLS)

    def test_register_overwrites_existing(self):
        registry = ToolRegistry(tools=[])
        t1 = ToolDef("foo", "First.", [])
        t2 = ToolDef("foo", "Second.", [])
        registry.register(t1)
        registry.register(t2)
        assert registry.get("foo").description == "Second."


class TestToolSelector:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_no_filter_returns_all(self):
        selector = ToolSelector(self.registry)
        assert set(selector.names()) == set(self.registry.names())

    def test_filter_to_subset(self):
        selector = ToolSelector(self.registry, allowed=["read_file", "write_file"])
        assert set(selector.names()) == {"read_file", "write_file"}

    def test_filter_excludes_others(self):
        selector = ToolSelector(self.registry, allowed=["read_file"])
        assert "write_file" not in selector.names()
        assert "run_shell" not in selector.names()

    def test_get_allowed_tool(self):
        selector = ToolSelector(self.registry, allowed=["read_file"])
        tool = selector.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_blocked_tool_returns_none(self):
        selector = ToolSelector(self.registry, allowed=["read_file"])
        assert selector.get("run_shell") is None

    def test_get_missing_tool_returns_none(self):
        selector = ToolSelector(self.registry)
        assert selector.get("does_not_exist") is None

    def test_select_returns_new_selector(self):
        selector = ToolSelector(self.registry)
        narrowed = selector.select(["read_file"])
        assert narrowed is not selector
        assert narrowed.names() == ["read_file"]

    def test_to_registry_contains_only_selected(self):
        selector = ToolSelector(self.registry, allowed=["read_file", "python_eval"])
        sub = selector.to_registry()
        assert set(sub.names()) == {"read_file", "python_eval"}

    def test_empty_allowed_list(self):
        selector = ToolSelector(self.registry, allowed=[])
        assert selector.available() == []
        assert selector.names() == []

    def test_available_with_unknown_names_ignored(self):
        selector = ToolSelector(self.registry, allowed=["read_file", "nonexistent"])
        names = selector.names()
        assert "read_file" in names
        assert "nonexistent" not in names
