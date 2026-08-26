# Tools Core

Read this file when the user wants to add function tools, organize toolsets, connect MCP servers, or use explicit common search tools.

## Add Tools to an Agent

Use `@agent.tool_plain` for pure functions and `@agent.tool` for tools that need `RunContext`.

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent('google-gla:gemini-3-flash-preview', deps_type=str)


@agent.tool_plain
def roll_dice() -> str:
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    return ctx.deps
```

Use `Tool(fn)` when tools are defined outside the agent file or shared between agents.

## Choosing a Tool Registration Method

Default choices:

- `@agent.tool` when the tool needs deps, usage, retry count, or message history
- `@agent.tool_plain` when the tool is a plain function
- `Tool(...)` in `tools=[...]` when the tool should be reusable across agents
- `FunctionToolset` when multiple related tools should be managed as a group

## Organize or Restrict Which Tools an Agent Can Use

Use toolsets when the user has multiple related tools or wants cross-cutting behavior applied to a group.

Examples:

- `FunctionToolset` for a bundle of Python tools
- MCP servers, which are themselves toolsets
- wrapper toolsets for approval, deferred loading, or other cross-cutting behavior

## Access Usage Stats, Message History, or Retry Count in Tools

Route this to `@agent.tool` with `RunContext`.

Useful `RunContext` fields include:

- `ctx.deps`
- `ctx.usage`
- `ctx.messages`
- `ctx.retry`

## Use MCP Servers

Attach an MCP server as a toolset on the agent.

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('python', args=['mcp_server.py'], timeout=10)
agent = Agent('openai:gpt-5.2', toolsets=[server])


async def main():
    async with agent:
        result = await agent.run('What is the weather in Paris?')
        print(result.output)
```

Default transport choices:

- `MCPServerStdio` for local subprocess servers
- `MCPServerStreamableHTTP` for HTTP servers

`MCPServerSSE` still exists, but Streamable HTTP is the better default.

## Search with DuckDuckGo, Tavily, or Exa

Use common tools when the user wants explicit search tools rather than provider-adaptive capabilities.

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[duckduckgo_search_tool()],
    instructions='Search DuckDuckGo for the given query and return the results.',
)
```

Good default split:

- use `WebSearch()` capability when the user wants model-agnostic search with builtin fallback
- use `duckduckgo_search_tool()` / Tavily / Exa when the user explicitly wants those engines as tools
