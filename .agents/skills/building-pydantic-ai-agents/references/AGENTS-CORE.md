# Agents Core

Read this file when the user needs the core `Agent` workflow: creating agents, choosing output types, using dependencies, defining specs, selecting models, or choosing how to run/stream an agent.

## Create a Basic Agent

```python
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

## Structured Output with Pydantic Models

Use `output_type=MyModel` when the model should return validated structured data.

```python
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-3-flash-preview', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
```

If the user is choosing between output modes:

- `output_type=str` for plain text
- `output_type=MyModel` for structured output
- `TextOutput` for custom text parsing
- `NativeOutput` or `ToolOutput` when they need explicit output-mode control

## Dependency Injection

Use `deps_type=...` plus `RunContext[...]` when tools or instructions need app state.

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5.2', deps_type=str)


@agent.instructions
def add_user_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."
```

Use `@agent.tool` when the tool needs `RunContext`. Use `@agent.tool_plain` when it does not.

## Define Agents Declaratively with Specs

Use YAML or JSON specs when configuration should live outside Python code.

```yaml
model: anthropic:claude-opus-4-6
instructions: "You are helping {{user_name}} with research."
capabilities:
  - WebSearch
  - Thinking:
      effort: high
```

```python
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_file('agent.yaml', deps_type=UserContext)
result = agent.run_sync('Find recent papers on AI safety', deps=UserContext(user_name='Alice'))
```

Template strings are part of the spec flow, so route template-string questions here too.

## Choose or Configure Models

Model strings use the `"provider:model-name"` format.

Examples:

- `openai:gpt-5.2`
- `anthropic:claude-sonnet-4-6`
- `google-gla:gemini-3-pro-preview`

Use a model instance instead of a string when the user needs provider-specific constructor arguments.

## Run Methods and Streaming

Pick a run method based on the interaction pattern:

- `run()` for async runs that complete normally
- `run_sync()` for synchronous scripts and notebooks
- `run_stream()` for streaming final output
- `run_stream_sync()` for sync streaming
- `run_stream_events()` when the caller needs the typed event stream directly
- `iter()` when the caller needs step-by-step control over the agent loop

Use `event_stream_handler=` with `run()` or `run_stream()` when the user wants progress updates without manually consuming the event stream:

```python
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, RunContext

agent = Agent('openai:gpt-5.2')


async def stream_handler(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]):
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            print(f'Calling {event.part.tool_name}...')


async def main():
    await agent.run('Do the task', event_stream_handler=stream_handler)
```

## Handle Provider Failures

Use `FallbackModel` when the user wants automatic provider or model failover.

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

fallback = FallbackModel(
    OpenAIChatModel('gpt-5.2'),
    AnthropicModel('claude-sonnet-4-6'),
)

agent = Agent(fallback)
```

Good defaults:

- primary expensive/strong model, cheaper fallback for resilience
- same prompt/output contract across both models
- per-model settings only when the user actually needs them
