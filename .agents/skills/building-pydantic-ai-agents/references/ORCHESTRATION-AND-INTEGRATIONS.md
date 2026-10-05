# Orchestration and Integrations

Read this file when the user wants multi-agent coordination, graphs, direct model calls, A2A, durable execution, embeddings, evals, or third-party integrations.

## Coordinate Multiple Agents

Use agent delegation when one agent should call another and return the result.

```python
from pydantic_ai import Agent, RunContext

parent = Agent('openai:gpt-5.2')
researcher = Agent('openai:gpt-5.2', output_type=str)


@parent.tool
async def research(ctx: RunContext[None], topic: str) -> str:
    result = await researcher.run(f'Research: {topic}', usage=ctx.usage)
    return result.output
```

Good split:

- delegation via tools when the parent keeps control
- output functions or programmatic hand-off when control should move elsewhere

## Build Multi-Step Workflows with Graphs

Use `pydantic_graph` when the workflow is a state machine rather than a single agent loop.

```python
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class FirstNode(BaseNode[None, None, int]):
    value: int

    async def run(self, ctx: GraphRunContext) -> 'SecondNode | End[int]':
        if self.value >= 5:
            return End(self.value)
        return SecondNode(self.value + 1)


@dataclass
class SecondNode(BaseNode):
    value: int

    async def run(self, ctx: GraphRunContext) -> FirstNode:
        return FirstNode(self.value)


graph = Graph(nodes=[FirstNode, SecondNode])
result = graph.run_sync(FirstNode(0))
```

## Call the Model Without Using an Agent

Use the direct API when the user wants a single model request without agent orchestration.

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

response = model_request_sync(
    'openai:gpt-5.2',
    [ModelRequest.user_text_prompt('Summarize this in one sentence.')],
)
```

Reach for this when there is no need for tools, retries, or agent loop state.

## Expose Agents as HTTP Servers (A2A)

Use `agent.to_a2a()` when the agent should be exposed as an ASGI app that speaks the A2A protocol.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')
app = agent.to_a2a()
```

## Use Durable Execution

Use the durable execution integrations when the run must survive crashes, retries, or long-lived workflows.

Temporal entry points:

- `TemporalAgent`
- `PydanticAIWorkflow`
- `PydanticAIPlugin`

There are parallel integrations for DBOS and Prefect.

## Use Embeddings for RAG

Use `Embedder(...)` for query/document embeddings when the user is building retrieval or semantic search.

```python
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')
```

## Use LangChain or ACI.dev Tools

Third-party integrations to reach for:

- `tool_from_langchain`
- `LangChainToolset`
- `tool_from_aci`
- `ACIToolset`

Use these when the user explicitly wants those ecosystems instead of native Pydantic AI tools.

## Systematically Verify Agent Behavior with Evals

Use `pydantic_evals` when the user wants repeatable evaluation datasets and evaluators rather than ad hoc tests.

Common entry points:

- `Case`
- `Dataset`
- evaluator classes from `pydantic_evals.evaluators`

## Build Custom Toolsets, Models, or Agents

Extensibility entry points:

- `AbstractToolset` / `WrapperToolset`
- `Model` / `WrapperModel`
- `AbstractAgent` / `WrapperAgent`
- `AbstractCapability`

Reach for these only when the built-in primitives are genuinely insufficient.
