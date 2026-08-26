# Architecture and Decision Guide

Detailed decision trees, comparison tables, and architecture overview for Pydantic AI.

## Contents

- [Task-Family References](#task-family-references)
- [Decision Trees](#decision-trees)
  - [Choosing a Tool Registration Method](#choosing-a-tool-registration-method)
  - [Choosing an Output Mode](#choosing-an-output-mode)
  - [Choosing a Multi-Agent Pattern](#choosing-a-multi-agent-pattern)
  - [Choosing How to Extend Agent Behavior](#choosing-how-to-extend-agent-behavior)
  - [Choosing a Capability](#choosing-a-capability)
  - [Choosing a Testing Approach](#choosing-a-testing-approach)
- [Comparison Tables](#comparison-tables)
  - [Output Mode Comparison](#output-mode-comparison)
  - [Model Provider Prefixes](#model-provider-prefixes)
  - [Tool Decorator Comparison](#tool-decorator-comparison)
  - [Built-in Capabilities](#built-in-capabilities)
  - [When to Use Each Agent Method](#when-to-use-each-agent-method)
- [Architecture Overview](#architecture-overview)

## Task-Family References

Use this file for comparisons and abstraction choices.

If the user already knows what they want to do, load the narrower task guide instead:

- [AGENTS-CORE.md](./AGENTS-CORE.md)
- [CAPABILITIES-AND-HOOKS.md](./CAPABILITIES-AND-HOOKS.md)
- [TOOLS-CORE.md](./TOOLS-CORE.md)
- [BUILTIN-TOOLS.md](./BUILTIN-TOOLS.md)
- [TOOLS-ADVANCED.md](./TOOLS-ADVANCED.md)
- [INPUT-AND-HISTORY.md](./INPUT-AND-HISTORY.md)
- [TESTING-AND-DEBUGGING.md](./TESTING-AND-DEBUGGING.md)
- [ORCHESTRATION-AND-INTEGRATIONS.md](./ORCHESTRATION-AND-INTEGRATIONS.md)

## Decision Trees

### Choosing a Tool Registration Method

```
Need RunContext (deps, usage, messages)?
├── Yes → Use @agent.tool
└── No → Pure function, no context needed?
    ├── Yes → Use @agent.tool_plain
    └── Tools defined outside agent file?
        ├── Yes → Use tools=[Tool(...)] in constructor
        └── Dynamic tools based on context?
            ├── Yes → Use ToolPrepareFunc
            └── Multiple related tools as a group?
                └── Yes → Use FunctionToolset
```

### Choosing an Output Mode

```
Need structured data with Pydantic validation?
├── Yes → Does provider support native JSON mode?
│   ├── Yes, and you want it → Use NativeOutput(MyModel)
│   └── No, or prefer consistency → Use ToolOutput(MyModel) [default]
└── No → Need custom parsing logic?
    ├── Yes → Use TextOutput(parser_fn)
    └── No → Just plain text?
        └── Yes → Use output_type=str [default]

Dynamic schema at runtime?
└── Yes → Use StructuredDict(json_schema)
```

### Choosing a Multi-Agent Pattern

```
Child agent returns result to parent?
├── Yes → Use agent delegation via tools
└── No → Permanent hand-off to specialist?
    ├── Yes → Use output functions
    └── Application code between agents?
        ├── Yes → Use programmatic hand-off
        └── Complex state machine?
            └── Yes → Use Graph-based control
```

### Choosing How to Extend Agent Behavior

```
Need reusable behavior across agents (tools + hooks + instructions)?
├── Yes → Build a custom capability (subclass AbstractCapability)
└── No → Just intercepting lifecycle events?
    ├── Yes → Complex interception needing tools/instructions too?
    │   ├── Yes → Subclass AbstractCapability
    │   └── No → Use Hooks capability with decorators
    └── No → Defining agents from config files?
        ├── Yes → Use Agent.from_file() with YAML/JSON specs
        └── No → Just adding tools?
            ├── Yes → Use @agent.tool or Toolset
            └── Pass args directly to Agent constructor
```

### Choosing a Capability

```
Need model thinking/reasoning?
├── Yes → Use Thinking(effort='high')
└── Need web search?
    ├── Yes → Use WebSearch() (auto-fallback to local)
    └── Need URL fetching?
        ├── Yes → Use WebFetch()
        └── Need MCP servers?
            ├── Yes → Use MCP()
            └── Need lifecycle hooks only?
                ├── Yes → Use Hooks()
                └── Need to filter/modify tool defs per step?
                    └── Yes → Use PrepareTools()
```

### Choosing a Testing Approach

```
Need deterministic, fast tests?
├── Yes → Use TestModel with agent.override()
└── Need specific tool call behavior?
    ├── Yes → Use FunctionModel
    └── Testing against real API (integration)?
        └── Yes → Use pytest-recording with VCR cassettes
```

## Comparison Tables

### Output Mode Comparison

| Scenario | Mode |
|----------|------|
| Need structured data and want maximum provider compatibility | `ToolOutput` (default) — works with all providers, supports streaming |
| Want the provider to natively enforce JSON schema compliance | `NativeOutput` — OpenAI, Anthropic, Google only; limited streaming |
| Provider doesn't support tools or JSON mode | `PromptedOutput` — works everywhere as a fallback |
| LLM returns non-JSON structured text (markdown, YAML, domain-specific) | `TextOutput` — custom parsing function |

### Model Provider Prefixes

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `openai:gpt-5.2` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-6` |
| Google (AI Studio) | `google-gla:` | `google-gla:gemini-3-pro-preview` |
| Google (Vertex) | `google-vertex:` | `google-vertex:gemini-3-pro-preview` |
| Groq | `groq:` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `cohere:command-r-plus-08-2024` |
| AWS Bedrock | `bedrock:` | `bedrock:anthropic.claude-sonnet-4-6` |
| Azure | `azure:` | `azure:gpt-5.2` |
| OpenRouter | `openrouter:` | `openrouter:anthropic/claude-sonnet-4-6` |
| xAI | `xai:` | `xai:grok-3` |
| DeepSeek | `deepseek:` | `deepseek:deepseek-chat` |
| Fireworks | `fireworks:` | `fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct` |
| Together | `together:` | `together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| Ollama (local) | `ollama:` | `ollama:llama3.2` |
| GitHub Models | `github:` | `github:openai/gpt-5.2` |
| Hugging Face | `huggingface:` | `huggingface:meta-llama/Llama-3.3-70B-Instruct` |
| Cerebras | `cerebras:` | `cerebras:llama-4-scout-17b-16e-instruct` |
| Heroku | `heroku:` | `heroku:claude-sonnet-4-6` |

**Additional prefixes:** `litellm:`, `nebius:`, `ovhcloud:`, `alibaba:`, `sambanova:`, `vercel:`, `outlines:`, `moonshotai:`. For truly custom providers, subclass `Model` or use `OpenAIChatModel` with a custom `base_url`.

### Tool Decorator Comparison

| Scenario | Decorator |
|----------|-----------|
| Tool needs access to deps, usage stats, messages, or retry info | `@agent.tool` — `RunContext` as required first param |
| Pure function, no agent context needed | `@agent.tool_plain` |
| Tools defined in a separate module or shared across agents | `Tool(fn)` — pass to agent constructor via `tools=[...]` |

### Built-in Capabilities

| Capability | What it provides | Usable in YAML Specs |
|---|---|:---:|
| `Thinking` | Model thinking/reasoning at configurable effort | Yes |
| `Hooks` | Decorator-based lifecycle hook registration | No |
| `WebSearch` | Web search — builtin when supported, local fallback | Yes |
| `WebFetch` | URL fetching — builtin when supported, custom fallback | Yes |
| `ImageGeneration` | Image generation — builtin when supported, custom fallback | Yes |
| `MCP` | MCP server — builtin when supported, direct connection | Yes |
| `PrepareTools` | Filters or modifies tool definitions per step | No |
| `PrefixTools` | Wraps a capability and prefixes its tool names | Yes |
| `BuiltinTool` | Registers a builtin tool with the agent | Yes |
| `Toolset` | Wraps an `AbstractToolset` | No |
| `HistoryProcessor` | Wraps a history processor function | No |

### When to Use Each Agent Method

| Scenario | Method |
|----------|--------|
| Building a chatbot or assistant that shows tool calls, progress, and output in real-time | `agent.run(event_stream_handler=...)` — streams all events while running to completion |
| Running an autonomous agent, batch job, or background task | `agent.run()` |
| Writing a CLI tool, script, or Jupyter notebook (no async) | `agent.run_sync()` |
| Streaming final text word-by-word to a UI | `agent.run_stream()` |
| Synchronous streaming for CLI tools or scripts (no async) | `agent.run_stream_sync()` |
| Receiving an async iterable of typed events (tool calls, results, final output) | `agent.run_stream_events()` |
| Inspecting or modifying state between agent steps, human-in-the-loop approval | `agent.iter()` |

See [Run Methods and Streaming](./AGENTS-CORE.md#run-methods-and-streaming) for `event_stream_handler` details.

## Architecture Overview

**Agent execution flow:**
`Agent.run()` → `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → (loop or end)

**Key generic types:**

- `Agent[AgentDepsT, OutputDataT]` — dependency type + output type
- `RunContext[AgentDepsT]` — available in tools and system prompts
- `AbstractCapability[AgentDepsT]` — base class for reusable behavior bundles

**Agent construction:**

- **Python:** `Agent(model, instructions=..., tools=..., capabilities=...)`
- **Declarative:** `Agent.from_file('agent.yaml')` or `Agent.from_spec({...})`

**Capabilities** are the primary extension point — they bundle tools, lifecycle hooks, instructions, and model settings into reusable units. Built-in capabilities include `Thinking`, `WebSearch`, `WebFetch`, `Hooks`, `MCP`, and more.

**Lifecycle hooks** (via `Hooks` or `AbstractCapability`) intercept every stage: `before_run` → `before_model_request` → `before_tool_execute` → `after_tool_execute` → `after_model_request` → `after_run`

**Model string format:** `"provider:model-name"` (e.g., `"openai:gpt-5.2"`, `"anthropic:claude-sonnet-4-6"`, `"google-gla:gemini-3-pro-preview"`)

**Output modes:**

- `ToolOutput` — structured data via tool calls (default for Pydantic models)
- `NativeOutput` — provider-specific structured output
- `PromptedOutput` — prompt-based structured extraction
- `TextOutput` — plain text responses
