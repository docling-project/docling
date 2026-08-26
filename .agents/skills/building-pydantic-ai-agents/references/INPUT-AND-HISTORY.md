# Input and History

Read this file when the user wants multimodal input, message history, or context trimming.

## Send Images, Audio, Video, or Documents to the Model

Pass multimodal content as a list mixing text with `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, or `BinaryContent`.

```python
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-5.2')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://example.com/logo.png'),
    ]
)
print(result.output)
```

Use `BinaryContent(...)` when the asset is already in memory instead of at a URL.

Not every model supports every input type. Keep provider expectations in mind when the user chooses a specific model.

## Work with Message History

Use `message_history=` to continue a conversation across runs.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.output)
```

Important distinctions:

- `new_messages()` returns only the current run
- `all_messages()` returns the full history accumulated so far
- when `message_history` is non-empty, Pydantic AI assumes the history already carries the system prompt

## Manage Context Size

Use `history_processors=[...]` to trim or rewrite message history before each model request.

```python
from pydantic_ai import Agent, ModelMessage


async def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-10:] if len(messages) > 10 else messages


agent = Agent('openai:gpt-5.2', history_processors=[keep_recent])
```

Good uses:

- trimming long conversations
- removing PII before provider calls
- summarizing old messages
- applying app-specific history policies
