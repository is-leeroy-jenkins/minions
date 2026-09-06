# Minions

![Minions project](resources/images/minions-project.png)

Minions provides small, provider-specific workflow agents. Guro supplies reusable instructions,
Fonky supplies provider-native tools, and Minions owns the execution workflow. A workflow never
translates or mixes tools between providers.

## Design

Each flat provider module exports its provider-local `Minion` and the same concrete workflow
implementations:

| Concrete implementation | Workflow purpose |
|---|---|
| `DataMinion` | Data analysis, transformation, statistics, and visualization |
| `ResearchMinion` | Source discovery, retrieval, synthesis, and evidence review |
| `CodingMinion` | Software engineering, debugging, testing, and code review |
| `WritingMinion` | Drafting, editing, summarization, and documentation |
| `PlanningMinion` | Project, task, process, and implementation planning |
| `ReviewMinion` | Evaluation, compliance, quality assurance, and red-team review |

| Provider module | Native design | Complete execution |
|---|---|---|
| `minions.gpt` | `Minion(agents.Agent)` | sync, async, stream |
| `minions.gemini` | `Minion(google.adk.Agent)` | sync, async, SSE stream |
| `minions.grok` | wraps xAI sync/async chats | sync, async, stream |
| `minions.claude` | wraps Anthropic tool runners | sync, async, stream |
| `minions.mistral` | wraps a Mistral remote Agent | sync, async, stream |

Every concrete implementation inherits its provider's `Minion` and receives its behavior from the
selected model, Guro instructions, provider-native tools, and execution limits. There is no factory
function, registry, or cross-provider abstract base class. Tools are optional for every provider.
OpenAI, Gemini, and Claude execute provider-native tools directly. Grok and Mistral accept provider
schemas and matching local functions; when supplied, the two name sets must match exactly. Their
streaming methods execute requested tools and resume streaming until the model finishes.

Select the concrete implementation that describes the workflow and pair it with the appropriate
Guro instructions:

```python
from guro import instructions
from minions.gpt import CodingMinion, ResearchMinion


coder = CodingMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'SENIOR_ENGINEER' ),
)
researcher = ResearchMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'DEEP_RESEARCH_AGENT' ),
)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Set the API key for the provider being used.

## OpenAI

```python
from fonky.gpt import tools
from guro import instructions
from minions.gpt import DataMinion


minion = DataMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.fetch_wikipedia, tools.load_csv ],
)
result = minion.run( 'Analyze the available evidence.' )
```

## Gemini

```python
from fonky.gemini import tools
from guro import instructions
from minions.gemini import DataMinion


minion = DataMinion(
    model='gemini-2.5-flash',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.fetch_wikipedia, tools.load_csv ],
)
result = minion.run( 'Analyze the available evidence.' )
```

## Grok

```python
from fonky.grok import tools
from guro import instructions
from minions.grok import DataMinion


minion = DataMinion(
    model='grok-4.5',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.wikipedia_fetch_tool, tools.csv_tool ],
    functions=[ tools.fetch_wikipedia, tools.load_csv ],
)
result = minion.run( 'Analyze the available evidence.' )
```

## Claude

```python
from fonky.claude import tools
from guro import instructions
from minions.claude import DataMinion


minion = DataMinion(
    model='claude-sonnet-4-6',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.fetch_wikipedia, tools.load_csv ],
)
result = minion.run( 'Analyze the available evidence.' )
```

## Mistral

```python
from fonky.mistral import tools
from guro import instructions
from minions.mistral import DataMinion


minion = DataMinion(
    model='mistral-medium-latest',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.wikipedia_fetch_tool, tools.csv_tool ],
    functions=[ tools.fetch_wikipedia, tools.load_csv ],
)
result = minion.run( 'Analyze the available evidence.' )
```

A tool-free workflow omits both optional arguments:

```python
minion = DataMinion(
    model='provider-model',
    instructions='Answer directly without tools.',
)
```

## Async and streaming

```python
result = await minion.run_async( 'Analyze the supplied data.' )
```

OpenAI returns its native `RunResultStreaming`. Gemini and Grok expose async event iterators.
Claude returns its native async streaming tool runner. Mistral exposes a synchronous iterator of
native `CompletionEvent` objects. Every pathway owns tool execution through the final response.

## Development

```powershell
python -m pip install -e .
python -m pip install pytest pytest-asyncio build
python -m pytest
python -m build
```

The test suite uses provider-native types with mocked network boundaries, so it consumes no API
credits.
