New AI trends focus on Agentic AI, an artificial intelligence system that can accomplish a specific goal with limited supervision.
Agents can act autonomously to understand, plan, and execute a specific task.

To address the integration problem, the [Model Context Protocol](https://modelcontextprotocol.io) (MCP) emerges as a popular standard for connecting AI applications to external tools.

## Docling MCP

Docling supports the development of AI agents by providing an MCP Server. It allows you to experiment with document processing in different MCP Clients. Adding [Docling MCP](https://github.com/docling-project/docling-mcp) in your favorite client is usually as simple as adding the following entry in the configuration file:

```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp",
        "docling-mcp-server"
      ]
    }
  }
}
```

When using [Claude on your desktop](https://claude.ai/download), just edit the config file `claude_desktop_config.json` with the snippet above or the example provided [here](https://github.com/docling-project/docling-mcp/blob/main/docs/integrations/claude_desktop_config.json).

In **[LM Studio](https://lmstudio.ai/)**, edit the `mcp.json` file with the appropriate section or simply click on the button below for a direct install.

[![Add MCP Server docling to LM Studio](https://files.lmstudio.ai/deeplink/mcp-install-light.svg)](https://lmstudio.ai/install-mcp?name=docling&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyItLWZyb209ZG9jbGluZy1tY3AiLCJkb2NsaW5nLW1jcC1zZXJ2ZXIiXX0%3D)


Docling MCP also provides tools specific for some applications and frameworks. See the [Docling MCP](https://github.com/docling-project/docling-mcp) Server repository for more details. You will find examples of building agents powered by Docling capabilities and leveraging frameworks like [LlamaIndex](https://www.llamaindex.ai/), [Llama Stack](https://github.com/llamastack/llama-stack), [Pydantic AI](https://ai.pydantic.dev/), or [smolagents](https://github.com/huggingface/smolagents).
