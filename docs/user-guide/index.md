# User Guide

Use this guide to select a provider, choose a workflow class, attach provider-compatible tools,
and run or stream the result.

## Choose a path

| Goal | Guide |
| --- | --- |
| Install, authenticate, and run a first Minion | [Getting Started](getting-started.md) |
| Add hosted search, code, file, or image tools | [Provider Tools](provider-tools.md) |
| Use async execution and provider-native streaming | [Execution and Streaming](execution.md) |
| Create a reusable workflow specialization | [Custom Minions](custom-minions.md) |

## The core pattern

```python
from minions.gpt import ResearchMinion


minion = ResearchMinion(
    model='gpt-5.6-sol',
    instructions='Research the question and distinguish evidence from conclusions.',
)

result = minion.run( 'Explain the requested subject.' )
print( result.final_output )
```

Tools are optional. When tools are used, import them from the same provider pathway as the
Minion.

!!! important "Keep a workflow provider-pure"
    Do not pass an OpenAI tool to a Gemini Minion, a Gemini tool to a Claude Minion, or otherwise
    mix provider contracts. Minions preserves provider-native behavior rather than translating
    schemas.
