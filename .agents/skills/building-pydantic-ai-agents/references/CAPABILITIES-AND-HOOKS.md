# Capabilities and Hooks

Read this file when the user wants reusable agent behavior, provider-adaptive tools, or lifecycle interception.

## Add Capabilities to an Agent

Capabilities bundle reusable behavior and compose automatically.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

Provider-adaptive capabilities to reach for first:

- `Thinking`
- `WebSearch`
- `WebFetch`
- `ImageGeneration`
- `MCP`

Use capabilities when the user wants behavior that should survive model/provider changes.

## Enable Thinking Across Providers

Use the unified `Thinking` capability or the `thinking` model setting.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-opus-4-6', capabilities=[Thinking(effort='high')])
agent = Agent('anthropic:claude-opus-4-6', model_settings={'thinking': 'high'})
```

Supported effort values:

- `True`
- `False`
- `'minimal'`
- `'low'`
- `'medium'`
- `'high'`
- `'xhigh'`

## Intercept Agent Lifecycle with Hooks

Use `Hooks` for decorator-based lifecycle interception.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages')
    return request_context


@hooks.on.before_tool_execute(tools=['send_email'])
async def audit_tool(ctx, *, call, tool_def, args):
    print(f'Executing {call.tool_name}')
    return args


agent = Agent('openai:gpt-5.2', capabilities=[hooks])
```

Important hook families:

- run-level hooks
- node-level hooks
- model-request hooks
- tool-validation hooks
- tool-execution hooks
- event-stream hooks

Use hooks when the user wants observability, auditing, or light interception without adding a new abstraction.

## Build a Custom Capability

Subclass `AbstractCapability` when the user wants reusable behavior that combines tools, hooks, instructions, or model settings into one package.

Reach for a custom capability when:

- the same bundle should be reused across multiple agents
- `Hooks` alone is not enough
- the behavior should be installable or declarative

Keep custom capabilities focused. If the user only needs one tool or one hook, do not introduce a capability.
