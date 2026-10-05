# Testing and Debugging

Read this file when the user wants deterministic tests, custom test models, request inspection, or runtime debugging.

## Test Agent Behavior

Use `TestModel` for fast deterministic tests and `FunctionModel` for custom response logic.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5.2')

with agent.override(model=TestModel()):
    result = agent.run_sync('test prompt')
    assert result.output == 'success (no tool calls)'
```

```python
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

agent = Agent('openai:gpt-5.2')


def custom_model(messages, info):
    return ModelResponse(parts=[TextPart(content='mocked response')])


with agent.override(model=FunctionModel(custom_model)):
    result = agent.run_sync('test prompt')
```

Default split:

- `TestModel` when you want automatic valid outputs
- `FunctionModel` when you need exact behavior for assertions, failures, or retries

## Debug a Failed Agent Run

Use `capture_run_messages()` when the user needs the exact request/response history that led to a failure.

```python
from pydantic_ai import Agent, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-5.2')

with capture_run_messages() as messages:
    try:
        agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior:
        print(messages)
```

Use this for in-process debugging. It is a better fit than broad logging when the user wants to inspect one failing run.

## Debug and Validate Agent Behavior

Use Logfire when the user wants observability across agent runs, tools, and model requests.

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)
```

Good uses:

- tracing tool calls
- validating what was sent to the provider
- understanding structured-output failures
- production observability
