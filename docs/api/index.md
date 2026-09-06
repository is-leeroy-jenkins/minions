# API Reference

The API reference is generated from the package's current type annotations and docstrings.

| Module | Scope |
| --- | --- |
| [`minions.gpt`](gpt.md) | OpenAI Agents SDK Minions and native tools |
| [`minions.gemini`](gemini.md) | Google ADK Minions and native tools |
| [`minions.grok`](grok.md) | xAI Minions, hosted tools, and local tool loops |
| [`minions.claude`](claude.md) | Anthropic Minions and native tool runners |
| [`minions.mistral`](mistral.md) | Mistral remote agents, hosted tools, and local tool loops |
| [`minions.config`](config.md) | Supported model identifiers and validation helper |

All provider modules export the same concrete Minion class family. Provider-specific tool and
result types remain visible in their signatures.
