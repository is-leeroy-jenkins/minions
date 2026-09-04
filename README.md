###### minions
![](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/images/minions-project.png)

___

A lightweight Python framework for reusable, provider-specific AI workflow agents.

The shared `Minion` class defines the common execution contract. Concrete agents are named for
the work they perform and live inside their provider namespace. Tools and agents used in one
workflow should come from the same provider.

## Current Implementation

The current milestone provides:

- The shared `minions.Minion` contract.
- The OpenAI `minions.gpt.DataAgent` workflow agent.
- Synchronous, asynchronous, and streamed OpenAI Agents SDK execution.
- Direct support for OpenAI-compatible tools, including `fonky.gpt.tools`.
- Provider-native execution results.


## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Configure the OpenAI credential before making live requests:

```powershell
$env:OPENAI_API_KEY = "..."
```

## OpenAI Data Workflow

```python
from fonky.gpt import tools
from guro import instructions
from minions.gpt import DataAgent


agent = DataAgent(
    name='Data Agent',
    model='gpt-5.6-terra',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
    max_turns=10,
)

result = agent.run( 'Analyze the available evidence and summarize the findings.' )
print( result.final_output )
```

Fonky tools are decorated tool objects. Pass them directly rather than calling them during agent
construction.

## Async Execution

```python
result = await agent.run_async( 'Analyze the supplied dataset.' )
```

## Streaming

```python
stream = agent.stream( 'Analyze the supplied dataset.' )

async for event in stream.stream_events( ):
    print( event )
```

## Package Boundaries

| Package   | Responsibility                    |
|-----------|-----------------------------------|
| `guro`    | Reusable instruction strings      |
| `fonky`   | Provider-compatible tools         |
| `minions` | Provider-specific workflow agents |

Minions does not convert tools between providers and does not normalize provider-native results.

## Development

Install the test dependencies and run the suite from the repository root:

```powershell
python -m pip install pytest pytest-asyncio
python -m pytest
```

Tests mock provider execution and do not consume API credits.

## Related Projects

- [fonky](https://github.com/is-leeroy-jenkins/fonky) - provider-compatible AI tools
- [guro](https://github.com/is-leeroy-jenkins/guro) - reusable instruction library
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
