---
name: building-pydantic-ai-agents
description: Build AI agents with Pydantic AI — tools, capabilities, structured output, streaming, testing, and multi-agent patterns. Use when the user mentions Pydantic AI, imports pydantic_ai, or asks to build an AI agent, add tools/capabilities, stream output, define agents from YAML, or test agent behavior.
license: MIT
compatibility: Requires Python 3.10+
metadata:
  version: "1.1.0"
  author: pydantic
---

# Building AI Agents with Pydantic AI

Pydantic AI is a Python agent framework for building production-grade Generative AI applications.
This skill provides patterns, architecture guidance, and tested code examples for building applications with Pydantic AI.

## When to Use This Skill

Invoke this skill when:
- User asks to build an AI agent, create an LLM-powered app, or mentions Pydantic AI
- User wants to add tools, capabilities (thinking, web search), or structured output to an agent
- User asks to define agents from YAML/JSON specs or use template strings
- User wants to stream agent events, delegate between agents, or test agent behavior
- Code imports `pydantic_ai` or references Pydantic AI classes (`Agent`, `RunContext`, `Tool`)
- User asks about hooks, lifecycle interception, or agent observability with Logfire

Do **not** use this skill for:
- The Pydantic validation library alone (`pydantic`/`BaseModel` without agents)
- Other AI frameworks (LangChain, LlamaIndex, CrewAI, AutoGen)
- General Python development unrelated to AI agents

## Quick-Start Patterns

### Create a Basic Agent

```python
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

### Add Tools to an Agent

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-3-flash-preview',
    deps_type=str,
    instructions=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)


@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Anne')
print(dice_result.output)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

### Structured Output with Pydantic Models

```python
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-3-flash-preview', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> RunUsage(input_tokens=57, output_tokens=8, requests=1)
```

### Dependency Injection

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5.2',
    deps_type=str,
    instructions="Use the customer's name while replying to them.",
)


@agent.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

### Testing with TestModel

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('openai:gpt-5.2', instructions='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []
```

### Use Capabilities

Capabilities are reusable, composable units of agent behavior — bundling tools, hooks, instructions, and model settings.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

### Add Lifecycle Hooks

Use `Hooks` to intercept model requests, tool calls, and runs with decorators — no subclassing needed.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages')
    return request_context


agent = Agent('openai:gpt-5.2', capabilities=[hooks])
```

### Define Agent from YAML Spec

Use `Agent.from_file` to load agents from YAML or JSON — no Python agent construction code needed.

```python
from pydantic_ai import Agent

# agent.yaml:
# model: anthropic:claude-opus-4-6
# instructions: You are a helpful research assistant.
# capabilities:
#   - WebSearch
#   - Thinking:
#       effort: high

agent = Agent.from_file('agent.yaml')
```

## Task Routing Table

Load only the most relevant reference first. Read additional references only if the task spans multiple areas.

| I want to... | Reference |
|---|---|
| Create/configure agents, choose output types, use deps, define specs, or pick run methods | [Agents Core](./references/AGENTS-CORE.md) |
| Bundle reusable behavior or intercept lifecycle events | [Capabilities and Hooks](./references/CAPABILITIES-AND-HOOKS.md) |
| Add function tools, toolsets, MCP servers, or explicit search tools | [Tools Core](./references/TOOLS-CORE.md) |
| Use provider-native web search, web fetch, or code execution | [Built-in Tools](./references/BUILTIN-TOOLS.md) |
| Use advanced tool features such as approval, retries, `ToolReturn`, validators, timeouts, or tool search | [Tools Advanced](./references/TOOLS-ADVANCED.md) |
| Work with multimodal input, message history, or context trimming | [Input and History](./references/INPUT-AND-HISTORY.md) |
| Test or debug agent behavior | [Testing and Debugging](./references/TESTING-AND-DEBUGGING.md) |
| Coordinate multiple agents or build graph workflows | [Orchestration and Integrations](./references/ORCHESTRATION-AND-INTEGRATIONS.md#coordinate-multiple-agents) |
| Call the model directly, expose A2A, use durable execution, embeddings, evals, or third-party integrations | [Orchestration and Integrations](./references/ORCHESTRATION-AND-INTEGRATIONS.md) |
| Compare abstractions, output modes, decorators, or model-string patterns | [Architecture and Decision Guide](./references/ARCHITECTURE.md) |
| Follow an older link into `COMMON-TASKS.md` | [Task Reference Map](./references/COMMON-TASKS.md) |

## Architecture and Decisions

Load [Architecture and Decision Guide](./references/ARCHITECTURE.md) only when the user is choosing between abstractions or wants comparison tables and decision trees:

| Topic | What it covers |
|---|---|
| Decision Trees | Tool registration, output modes, multi-agent patterns, capabilities, testing approaches, extensibility |
| Comparison Tables | Output modes, model provider prefixes, tool decorators, built-in capabilities, agent methods |
| Architecture Overview | Execution flow, generic types, construction patterns, lifecycle hooks, model string format |

**Quick reference — model string format:** `"provider:model-name"` (e.g., `"openai:gpt-5.2"`, `"anthropic:claude-sonnet-4-6"`, `"google-gla:gemini-3-pro-preview"`)

**Quick reference — key agent methods:** `run()`, `run_sync()`, `run_stream()`, `run_stream_sync()`, `run_stream_events()`, `iter()`

## Key Practices

- **Python 3.10+** compatibility required
- **Observability**: Pydantic AI has first-class integration with Logfire for tracing agent runs, tool calls, and model requests. Add it with `logfire.instrument_pydantic_ai()`. For deeper HTTP-level visibility, `logfire.instrument_httpx(capture_all=True)` captures the exact payloads sent to model providers.
- **Testing**: Use `TestModel` for deterministic tests, `FunctionModel` for custom logic

## Common Gotchas

These are mistakes agents commonly make with Pydantic AI. Getting these wrong produces silent failures or confusing errors.

- **`@agent.tool` requires `RunContext` as first param**; `@agent.tool_plain` must **not** have it. Mixing these up causes runtime errors. Use `tool_plain` when you don't need deps, usage, or messages.
- **Model strings need the provider prefix**: `'openai:gpt-5.2'` not `'gpt-5.2'`. Without the prefix, Pydantic AI can't resolve the provider.
- **`TestModel` requires `agent.override()`**: Don't set `agent.model` directly. Always use the context manager: `with agent.override(model=TestModel()):`.
- **`str` in output_type allows plain text to end the run**: If your union includes `str` (or no `output_type` is set), the model can return plain text instead of structured output. Omit `str` from the union to force tool-based output.
- **Hook decorator names on `.on` don't repeat `on_`**: Use `hooks.on.run_error` and `hooks.on.model_request_error` — not `hooks.on.on_run_error`.
- **`history_processors` is plural**: The Agent parameter is `history_processors=[...]`, not `history_processor=`.

## Task-Family References

Load exactly one of these unless the task clearly spans multiple families:

| Task family | Reference |
|---|---|
| Core agent setup, output, deps, specs, models, run methods | [Agents Core](./references/AGENTS-CORE.md) |
| Capabilities, hooks, and reusable behavior | [Capabilities and Hooks](./references/CAPABILITIES-AND-HOOKS.md) |
| Function tools, toolsets, MCP, explicit search tools | [Tools Core](./references/TOOLS-CORE.md) |
| Provider-native builtin tools | [Built-in Tools](./references/BUILTIN-TOOLS.md) |
| Approval, retries, validators, timeouts, rich tool returns, deferred loading | [Tools Advanced](./references/TOOLS-ADVANCED.md) |
| Multimodal input, message history, history processors | [Input and History](./references/INPUT-AND-HISTORY.md) |
| Testing, request inspection, and Logfire debugging | [Testing and Debugging](./references/TESTING-AND-DEBUGGING.md) |
| Multi-agent patterns, graphs, direct API, A2A, durable execution, embeddings, evals, third-party integrations | [Orchestration and Integrations](./references/ORCHESTRATION-AND-INTEGRATIONS.md) |

Use [Task Reference Map](./references/COMMON-TASKS.md) only for compatibility with older links or when you need a pointer from an old section name to the new file.
