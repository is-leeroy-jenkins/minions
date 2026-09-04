###### minions
![](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/images/minions-project.png)

___

A lightweight Python framework for reusable, provider-specific AI workflow agents.

The shared `Minion` class defines the common execution contract. Concrete agents are named for
the work they perform and live inside their provider namespace. Tools and agents used in one
workflow come from the same provider.

## Provider Support

| Provider | Agent | Tool contract | Synchronous | Asynchronous | Streaming |
|----------|-------|---------------|-------------|--------------|-----------|
| OpenAI | `minions.gpt.DataAgent` | OpenAI Agents SDK tools | Yes | Yes | Yes |
| Google | `minions.gemini.DataAgent` | Google ADK callables | Yes | Yes | Yes |
| xAI | `minions.grok.DataAgent` | xAI schemas paired with callables | Yes | Yes | Yes |
| Anthropic | `minions.claude.DataAgent` | Anthropic `@beta_tool` objects | Yes | Yes | Yes |
| Mistral AI | `minions.mistral.DataAgent` | Python callables | Yes | Yes | Native stream |

Every implementation returns its provider-native result. Minions does not translate tools or
normalize results across providers.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Set the credential required by the workflow provider:

```powershell
$env:OPENAI_API_KEY = "..."
$env:GOOGLE_API_KEY = "..."
$env:XAI_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:MISTRAL_API_KEY = "..."
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

Fonky GPT tools are decorated OpenAI tool objects and are passed directly to `DataAgent`.

## Gemini Data Workflow

```python
from fonky.gemini import tools
from guro import instructions
from minions.gemini import DataAgent


agent = DataAgent(
    name='Data Agent',
    model='gemini-2.5-flash',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
    max_turns=10,
)

result = agent.run( 'Analyze the available evidence and summarize the findings.' )
```

Google ADK registers the Fonky Gemini callables and executes their tool calls within the runner.

## Grok Data Workflow

```python
from fonky.grok import tools
from guro import instructions
from minions.grok import DataAgent


agent = DataAgent(
    name='Data Agent',
    model='grok-4.5',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.wikipedia_fetch_tool,
        tools.csv_tool,
    ],
    functions=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
    max_turns=10,
)

result = agent.run( 'Analyze the available evidence and summarize the findings.' )
```

The xAI SDK uses separate provider schemas and executable functions. `DataAgent` requires an exact
name match, executes every requested callable, and submits each result with its tool-call ID.

## Claude Data Workflow

```python
from fonky.claude import tools
from guro import instructions
from minions.claude import DataAgent


agent = DataAgent(
    name='Data Agent',
    model='claude-sonnet-4-6',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
    max_turns=10,
    max_tokens=4096,
)

result = agent.run( 'Analyze the available evidence and summarize the findings.' )
```

Fonky Claude tools are Anthropic `@beta_tool` objects. The synchronous and asynchronous Anthropic
tool runners execute them automatically; the async adapter preserves each Fonky tool schema.

## Mistral Data Workflow

```python
from fonky.mistral import tools
from guro import instructions
from minions.mistral import DataAgent


agent = DataAgent(
    name='Data Agent',
    model='mistral-medium-latest',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
    max_turns=10,
)

result = agent.run( 'Analyze the available evidence and summarize the findings.' )
print( result.choices[ 0 ].message.content )
```

`DataAgent` creates Mistral function schemas and completes local tool handling for synchronous and
asynchronous workflows. Mistral streaming returns the provider-native stream; streamed function
calls remain caller-controlled.

## Async Execution

All five agents implement the shared asynchronous contract:

```python
result = await agent.run_async( 'Analyze the supplied dataset.' )
```

## Streaming

Streaming retains the native provider interface. OpenAI returns an Agents SDK streaming result.
Gemini and Grok return async iterators. Claude returns an Anthropic async streaming tool runner.
Mistral returns its native synchronous event stream.

## Package Boundaries

| Package | Responsibility |
|---------|----------------|
| `guro` | Reusable instruction strings |
| `fonky` | Provider-compatible tools |
| `minions` | Provider-specific workflow agents |

## Development

Install the project and test dependencies, then run the complete suite:

```powershell
python -m pip install -e .
python -m pip install pytest pytest-asyncio build
python -m pytest
python -m build
```

Tests mock provider execution and do not consume API credits.

## Related Projects

- [fonky](https://github.com/is-leeroy-jenkins/fonky) - provider-compatible AI tools
- [guro](https://github.com/is-leeroy-jenkins/guro) - reusable instruction library
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Google Agent Development Kit](https://google.github.io/adk-docs/)
- [xAI Python SDK](https://github.com/xai-org/xai-sdk-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Mistral AI Python SDK](https://github.com/mistralai/client-python)
