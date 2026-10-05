# Task Reference Map

This file exists as a compatibility index for older links into `COMMON-TASKS.md`.

Prefer the narrower task-family guides below so the agent loads only the material it needs:

- [AGENTS-CORE.md](./AGENTS-CORE.md) — agent creation, output, deps, specs, models, run methods
- [CAPABILITIES-AND-HOOKS.md](./CAPABILITIES-AND-HOOKS.md) — `Thinking`, `WebSearch`, `Hooks`, custom capabilities
- [TOOLS-CORE.md](./TOOLS-CORE.md) — `@agent.tool`, `Tool`, toolsets, MCP, common search tools
- [BUILTIN-TOOLS.md](./BUILTIN-TOOLS.md) — provider-native tools like `WebSearchTool` and `CodeExecutionTool`
- [TOOLS-ADVANCED.md](./TOOLS-ADVANCED.md) — approval, retries, `ToolReturn`, timeouts, validators, deferred loading
- [INPUT-AND-HISTORY.md](./INPUT-AND-HISTORY.md) — multimodal input, message history, history processors
- [TESTING-AND-DEBUGGING.md](./TESTING-AND-DEBUGGING.md) — `TestModel`, `FunctionModel`, `capture_run_messages`, Logfire
- [ORCHESTRATION-AND-INTEGRATIONS.md](./ORCHESTRATION-AND-INTEGRATIONS.md) — multi-agent patterns, graphs, A2A, direct API, durable execution, embeddings, evals, third-party tools

## Add Capabilities to an Agent

Read [Add Capabilities](./CAPABILITIES-AND-HOOKS.md#add-capabilities-to-an-agent).

## Intercept Agent Lifecycle with Hooks

Read [Intercept Agent Lifecycle with Hooks](./CAPABILITIES-AND-HOOKS.md#intercept-agent-lifecycle-with-hooks).

## Define Agents Declaratively with Specs

Read [Define Agents Declaratively with Specs](./AGENTS-CORE.md#define-agents-declaratively-with-specs).

## Enable Thinking Across Providers

Read [Enable Thinking Across Providers](./CAPABILITIES-AND-HOOKS.md#enable-thinking-across-providers).

## Use MCP Servers

Read [Use MCP Servers](./TOOLS-CORE.md#use-mcp-servers).

## Search with DuckDuckGo, Tavily, or Exa

Read [Search with DuckDuckGo, Tavily, or Exa](./TOOLS-CORE.md#search-with-duckduckgo-tavily-or-exa).

## Require Tool Approval (Human in the Loop)

Read [Require Tool Approval](./TOOLS-ADVANCED.md#require-tool-approval-human-in-the-loop).

## Send Images, Audio, Video, or Documents to the Model

Read [Send Images, Audio, Video, or Documents to the Model](./INPUT-AND-HISTORY.md#send-images-audio-video-or-documents-to-the-model).

## Manage Context Size

Read [Manage Context Size](./INPUT-AND-HISTORY.md#manage-context-size).

## Work with Message History

Read [Work with Message History](./INPUT-AND-HISTORY.md#work-with-message-history).

## Show Real-Time Progress

Read [Run Methods and Streaming](./AGENTS-CORE.md#run-methods-and-streaming).

## Handle Provider Failures

Read [Handle Provider Failures](./AGENTS-CORE.md#handle-provider-failures).

## Make an Agent Resilient with Retries

Read [Make an Agent Resilient with Retries](./TOOLS-ADVANCED.md#make-an-agent-resilient-with-retries).

## Debug a Failed Agent Run

Read [Debug a Failed Agent Run](./TESTING-AND-DEBUGGING.md#debug-a-failed-agent-run).

## Test Agent Behavior

Read [Test Agent Behavior](./TESTING-AND-DEBUGGING.md#test-agent-behavior).

## Coordinate Multiple Agents

Read [Coordinate Multiple Agents](./ORCHESTRATION-AND-INTEGRATIONS.md#coordinate-multiple-agents).

## Build Multi-Step Workflows with Graphs

Read [Build Multi-Step Workflows with Graphs](./ORCHESTRATION-AND-INTEGRATIONS.md#build-multi-step-workflows-with-graphs).

## Debug and Validate Agent Behavior

Read [Debug and Validate Agent Behavior](./TESTING-AND-DEBUGGING.md#debug-and-validate-agent-behavior).

## Advanced and Less Common Features

Read only the relevant section in [ORCHESTRATION-AND-INTEGRATIONS.md](./ORCHESTRATION-AND-INTEGRATIONS.md):

- [Direct API](./ORCHESTRATION-AND-INTEGRATIONS.md#call-the-model-without-using-an-agent)
- [A2A](./ORCHESTRATION-AND-INTEGRATIONS.md#expose-agents-as-http-servers-a2a)
- [Durable Execution](./ORCHESTRATION-AND-INTEGRATIONS.md#use-durable-execution)
- [Embeddings](./ORCHESTRATION-AND-INTEGRATIONS.md#use-embeddings-for-rag)
- [Third-Party Tools](./ORCHESTRATION-AND-INTEGRATIONS.md#use-langchain-or-acidev-tools)
- [Custom Extensibility](./ORCHESTRATION-AND-INTEGRATIONS.md#build-custom-toolsets-models-or-agents)
- [Evals](./ORCHESTRATION-AND-INTEGRATIONS.md#systematically-verify-agent-behavior-with-evals)

## Working with the Installed Pydantic AI Package

Prefer the checked-out repository or these bundled references first.

Inspect local package source only as a last resort when:

- the user asks about a symbol that is not covered in this skill
- the local environment already includes `pydantic_ai` and you need exact module layout
- you have no narrower offline reference available
