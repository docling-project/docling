# Built-in Tools

Read this file when the user wants provider-native tools such as web search, web fetch, code execution, memory, or file search.

Prefer capabilities like `WebSearch()` or `WebFetch()` when the user wants a provider-agnostic solution. Use built-in tools directly when they explicitly want provider-native behavior or provider-specific configuration.

## Give My Agent Web Search or Code Execution

Builtin tools are passed via `builtin_tools=[...]`.

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-5.2', builtin_tools=[WebSearchTool()])
result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
```

For OpenAI web search, use the Responses API model prefix (`openai-responses:`), not `openai:`.

## Built-in Tool Defaults

Reach for these when the provider supports them:

- `WebSearchTool`
- `WebFetchTool`
- `CodeExecutionTool`
- `ImageGenerationTool`
- `MemoryTool`
- `MCPServerTool`
- `FileSearchTool`

## Dynamic Built-in Tool Configuration

Prepare built-in tools from `RunContext` when configuration depends on the current user or request.

```python
from pydantic_ai import Agent, RunContext, WebSearchTool


async def prepared_web_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    if not ctx.deps.get('location'):
        return None
    return WebSearchTool(user_location={'city': ctx.deps['location']})


agent = Agent(
    'openai-responses:gpt-5.2',
    builtin_tools=[prepared_web_search],
    deps_type=dict,
)
```

## When to Use Built-in Tools vs Capabilities

Use built-in tools when:

- the user explicitly wants provider-native behavior
- the provider-specific configuration matters
- the user already picked a provider that supports the tool

Use capabilities when:

- the code should work across providers
- you want local fallback when builtin support is missing
- the user has not committed to a provider yet
