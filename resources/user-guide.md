![](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/images/minions-userguide.png)
___

Minions provides provider-native workflow agents for OpenAI, Google Gemini, xAI Grok,
Anthropic Claude, and Mistral. Select one provider module, choose a concrete `Minion` class,
and supply the model, instructions, and optional tools required by the workflow.

## Installation

Install Minions directly from GitHub:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install git+https://github.com/is-leeroy-jenkins/minions.git
```

Install the companion Guro and Fonky packages when the workflow uses their instructions or tools:

```powershell
python -m pip install git+https://github.com/is-leeroy-jenkins/guro.git
python -m pip install git+https://github.com/is-leeroy-jenkins/fonky.git
```

## Authentication

Set the environment variable for the selected provider. A workflow only requires credentials for
the provider it uses.

```powershell
$env:OPENAI_API_KEY = '...'
$env:GOOGLE_API_KEY = '...'
$env:XAI_API_KEY = '...'
$env:ANTHROPIC_API_KEY = '...'
$env:MISTRAL_API_KEY = '...'
```

Grok, Claude, and Mistral also accept an `api_key` argument. Prefer environment variables so keys
do not appear in source code.

## Select a Concrete Minion

Every provider module exports the same concrete class family.

| Workflow category | Concrete class |
|---|---|
| Research / Academic | `ResearchMinion` |
| Writing / Administrative | `WritingMinion` |
| Compliance / Legal / Budget | `ComplianceMinion` |
| Business / Finance / Marketing | `BusinessMinion` |
| Software Engineering | `CodingMinion` |
| Data Analytics | `DataMinion` |
| Data Governance | `GovernanceMinion` |
| Instruction / Training / Planning | `PlanningMinion` |
| Image Generation | `ImageGenerationMinion` |
| Image Analysis | `ImageAnalysisMinion` |
| Image Editing | `ImageEditingMinion` |
| Translation | `TranslationMinion` |
| Transcription | `TranscriptionMinion` |
| Speech | `SpeechMinion` |

The class identifies the workflow role. Its `instructions` argument supplies the specialized
behavior for a particular use case.

## Common Constructor Arguments

| Argument | Required | Default | Purpose |
|---|---:|---:|---|
| `model` | Yes | — | Provider model identifier |
| `instructions` | Yes | — | System instructions governing the workflow |
| `tools` | No | `None` | Tools compatible with the selected provider |
| `max_turns` | No | `10` | Maximum model turns or calls for one execution |
| `name` | No | Class default | Human-readable Minion name |

Grok and Mistral additionally accept `functions`. Claude additionally accepts `max_tokens`, which
defaults to `4096`.

## Example 1: Tool-Free OpenAI Writing Minion

Tools are optional. Use a tool-free Minion for summarization, drafting, classification, and other
workflows that do not require external data or actions.

```python
from minions.gpt import WritingMinion


minion = WritingMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Write concise technical documentation for Python developers. '
        'Use direct language, accurate terminology, and complete examples.'
    ),
)

result = minion.run(
    'Draft release notes for a library that adds provider-native search tools.'
)
print( result.final_output )
```

`run` returns the OpenAI Agents SDK `RunResult`. The Minion stores the same object in
`minion.result`.

## Example 2: OpenAI Research with Native Web and File Search

OpenAI-native tools are imported from `minions.gpt` and passed directly to the inherited
`agents.Agent` implementation.

```python
from minions.gpt import FileSearchTool, ResearchMinion, WebSearchTool


minion = ResearchMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Research the question using current web sources and the indexed documents. '
        'Distinguish sourced facts from conclusions.'
    ),
    tools=[
        WebSearchTool( ),
        FileSearchTool( vector_store_ids=[ 'vs_1234567890' ] ),
    ],
    max_turns=10,
)

result = minion.run(
    'Compare the current policy with the requirements in the indexed documents.'
)
print( result.final_output )
```

Replace `vs_1234567890` with an OpenAI vector-store identifier available to the configured API
key.

## Example 3: OpenAI Code Interpreter

Use `CodeInterpreterTool` for calculations, structured-data analysis, and generated files.

```python
from minions.gpt import CodeInterpreterTool, DataMinion


minion = DataMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Use Python to validate calculations. Report assumptions, methods, and results.'
    ),
    tools=[
        CodeInterpreterTool(
            tool_config={
                'type': 'code_interpreter',
                'container': { 'type': 'auto' },
            },
        ),
    ],
)

result = minion.run(
    'Calculate the monthly compound growth of 125000 at 4.2 percent annually for 36 months.'
)
print( result.final_output )
```

## Example 4: Gemini Research with Google Search

Gemini Minions inherit from `google.adk.Agent`. The module exports Google ADK native tools without
wrapping or translating them.

```python
from minions.gemini import ResearchMinion, google_search


minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions=(
        'Use Google Search to gather current information. '
        'Return a concise synthesis with source attribution.'
    ),
    tools=[ google_search ],
)

event = minion.run( 'Summarize the latest material developments in agent SDKs.' )

for part in event.content.parts:
    if part.text:
        print( part.text )
```

The synchronous and asynchronous Gemini methods return the final ADK `Event`. All emitted events
remain available in `minion.events`.

## Example 5: Gemini Enterprise Retrieval with Vertex AI Search

Use the complete Vertex AI data-store resource name assigned to the active Google Cloud project.

```python
from minions.gemini import ResearchMinion, VertexAiSearchTool


data_store_id = (
    'projects/example-project/locations/global/collections/default_collection/'
    'dataStores/policy-library'
)

minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions=(
        'Search the enterprise policy library. Cite the documents supporting each finding.'
    ),
    tools=[ VertexAiSearchTool( data_store_id=data_store_id ) ],
)

event = minion.run( 'Identify the approval and record-retention requirements.' )
print( event )
```

## Example 6: Grok with Provider-Hosted Tools

xAI-hosted tools do not require entries in `functions`. The provider executes them directly.

```python
from minions.grok import DataMinion, code_execution, web_search


minion = DataMinion(
    model='grok-4.5',
    instructions=(
        'Research current data, verify calculations with code, and explain the result.'
    ),
    tools=[
        web_search( ),
        code_execution( ),
    ],
)

response = minion.run(
    'Find the latest reported value, calculate its year-over-year change, and explain the trend.'
)
print( response )
```

Other xAI-hosted tools exported from `minions.grok` are `collections_search` and
`image_generation`.

## Example 7: Grok with a Local Function Tool

Client-executed Grok functions require two matching components:

1. An xAI tool schema in `tools`.
2. A callable with the identical name in `functions`.

```python
from xai_sdk.chat import tool

from minions.grok import DataMinion


def lookup_account( account_id: str ) -> dict[ str, str ]:
    """Return an account record from the application data source."""
    return {
        'account_id': account_id,
        'status': 'active',
        'owner': 'Program Office',
    }


lookup_account_schema = tool(
    name='lookup_account',
    description='Return one account record by account identifier.',
    parameters={
        'type': 'object',
        'properties': {
            'account_id': {
                'type': 'string',
                'description': 'Account identifier to retrieve.',
            },
        },
        'required': [ 'account_id' ],
    },
)

minion = DataMinion(
    model='grok-4.5',
    instructions='Use the account tool to answer account-status questions.',
    tools=[ lookup_account_schema ],
    functions=[ lookup_account ],
)

response = minion.run( 'Report the status of account A-1042.' )
print( response )
```

Minions rejects duplicate, missing, or extra function names before sending a request to xAI.

## Example 8: Claude with Server Tools

Anthropic server tools are typed provider objects. They may be combined in the same `tools`
sequence.

```python
from minions.claude import (
    BetaCodeExecutionTool20260521Param,
    BetaWebFetchTool20260318Param,
    BetaWebSearchTool20260318Param,
    ResearchMinion,
)


minion = ResearchMinion(
    model='claude-sonnet-4-6',
    instructions=(
        'Research the subject, inspect relevant pages, and use code when calculations are needed.'
    ),
    tools=[
        BetaWebSearchTool20260318Param(
            type='web_search_20260318',
            name='web_search',
        ),
        BetaWebFetchTool20260318Param(
            type='web_fetch_20260318',
            name='web_fetch',
        ),
        BetaCodeExecutionTool20260521Param(
            type='code_execution_20260521',
            name='code_execution',
        ),
    ],
    max_tokens=4096,
)

message = minion.run( 'Research the subject and calculate the requested comparison.' )
print( message.content )
```

Claude uses Anthropic's native tool runner through the complete tool lifecycle.

## Example 9: Claude with a Local Tool

Use `anthropic.beta_tool` for client-executed local functions. A synchronous local tool is adapted
automatically when `run_async` or `stream` is used.

```python
from anthropic import beta_tool

from minions.claude import ComplianceMinion


@beta_tool
def lookup_requirement( citation: str ) -> dict[ str, str ]:
    """Return a compliance requirement by citation."""
    return {
        'citation': citation,
        'requirement': 'Document the approval and retain supporting evidence.',
    }


minion = ComplianceMinion(
    model='claude-sonnet-4-6',
    instructions=(
        'Evaluate compliance questions against retrieved requirements. '
        'Do not invent missing requirements.'
    ),
    tools=[ lookup_requirement ],
)

message = minion.run( 'Evaluate compliance with requirement A-12.' )
print( message.content )
```

## Example 10: Mistral with Native Tools

Mistral Minions create a provider-native remote agent during initialization. Provider-hosted tools
do not require entries in `functions`.

```python
from minions.mistral import (
    CodeInterpreterTool,
    DocumentLibraryTool,
    ResearchMinion,
    WebSearchTool,
)


minion = ResearchMinion(
    model='mistral-medium-latest',
    instructions=(
        'Use current web information and the configured document library. '
        'Validate numerical comparisons with code.'
    ),
    tools=[
        WebSearchTool( ),
        CodeInterpreterTool( ),
        DocumentLibraryTool( library_ids=[ 'library-id' ] ),
    ],
)

response = minion.run( 'Compare the current information with the document library.' )
print( response.choices[ 0 ].message.content )
```

Replace `library-id` with a Mistral document-library identifier available to the configured API
key.

## Example 11: Mistral with a Local Function Tool

The Mistral function schema name and Python callable name must match exactly.

```python
from minions.mistral import BusinessMinion


def calculate_variance( planned: float, actual: float ) -> dict[ str, float ]:
    """Calculate the amount and percentage variance."""
    amount = actual - planned
    percentage = 0.0 if planned == 0 else amount / planned * 100
    return { 'amount': amount, 'percentage': percentage }


variance_schema = {
    'type': 'function',
    'function': {
        'name': 'calculate_variance',
        'description': 'Calculate actual-versus-planned variance.',
        'parameters': {
            'type': 'object',
            'properties': {
                'planned': { 'type': 'number' },
                'actual': { 'type': 'number' },
            },
            'required': [ 'planned', 'actual' ],
        },
    },
}

minion = BusinessMinion(
    model='mistral-medium-latest',
    instructions='Use the variance tool and explain the business impact.',
    tools=[ variance_schema ],
    functions=[ calculate_variance ],
)

response = minion.run( 'Compare an actual value of 117.5 with a plan of 100.' )
print( response.choices[ 0 ].message.content )
```

## Example 12: Guro Instructions and Fonky Tools

Use Guro for reusable instructions and the provider-matched Fonky module for reusable tools.

```python
from fonky.gpt import tools
from guro import instructions
from minions.gpt import DataMinion


minion = DataMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[
        tools.fetch_wikipedia,
        tools.load_csv,
    ],
)

result = minion.run(
    'Load the supplied data, obtain relevant background information, and summarize the findings.'
)
print( result.final_output )
```

Match the imports to the selected provider. For example, `minions.gemini` should receive tools
from `fonky.gemini`, not `fonky.gpt`.

## Asynchronous Execution

All providers expose `run_async`.

```python
import asyncio

from minions.gpt import ResearchMinion


async def main( ) -> None:
    minion = ResearchMinion(
        model='gpt-5.6-sol',
        instructions='Research the subject and return a concise summary.',
    )
    result = await minion.run_async( 'Research the requested subject.' )
    print( result.final_output )


if __name__ == '__main__':
    asyncio.run( main( ) )
```

Grok and Mistral accept synchronous or asynchronous local tools during asynchronous execution.
Synchronous functions are moved to a worker thread so they do not block the event loop.

## Streaming Execution

Streaming uses the native convention of each provider.

### OpenAI Streaming

```python
import asyncio

from minions.gpt import WritingMinion


async def main( ) -> None:
    minion = WritingMinion(
        model='gpt-5.6-sol',
        instructions='Write a concise technical response.',
    )
    stream = minion.stream( 'Explain the Minions provider boundary.' )

    async for event in stream.stream_events( ):
        print( event )


if __name__ == '__main__':
    asyncio.run( main( ) )
```

### Gemini Streaming

```python
import asyncio

from minions.gemini import ResearchMinion


async def main( ) -> None:
    minion = ResearchMinion(
        model='gemini-2.5-flash',
        instructions='Research the subject and report the findings.',
    )

    async for event in minion.stream( 'Research the requested subject.' ):
        print( event )

    print( minion.result )


if __name__ == '__main__':
    asyncio.run( main( ) )
```

### Grok Streaming

```python
import asyncio

from minions.grok import ResearchMinion


async def main( ) -> None:
    minion = ResearchMinion(
        model='grok-4.5',
        instructions='Research the subject and report the findings.',
    )

    async for response, chunk in minion.stream( 'Research the requested subject.' ):
        print( chunk )

    print( minion.result )


if __name__ == '__main__':
    asyncio.run( main( ) )
```

### Claude Streaming

Claude returns Anthropic's provider-managed asynchronous streaming tool runner.

```python
from minions.claude import ResearchMinion


minion = ResearchMinion(
    model='claude-sonnet-4-6',
    instructions='Research the subject and report the findings.',
)

runner = minion.stream( 'Research the requested subject.' )
```

Consume `runner` using the streaming interfaces provided by the installed Anthropic SDK version.
The runner continues through server and local tool calls automatically.

### Mistral Streaming

```python
from minions.mistral import ResearchMinion


minion = ResearchMinion(
    model='mistral-medium-latest',
    instructions='Research the subject and report the findings.',
)

for event in minion.stream( 'Research the requested subject.' ):
    print( event.data )
```

## Create a Specialized Minion

Create reusable specializations through ordinary inheritance. Override `minion_name` and keep the
provider implementation unchanged.

```python
from minions.gpt import ComplianceMinion


class BudgetMinion( ComplianceMinion ):
    """OpenAI Minion specialized for federal budget compliance."""

    minion_name: str = 'Budget Minion'


minion = BudgetMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Evaluate budget requests for purpose, time, and amount compliance. '
        'Identify unsupported conclusions and missing evidence.'
    ),
)

result = minion.run( 'Review the supplied funding justification.' )
print( result.final_output )
```

## Override the Display Name

Use `name` for a one-off display-name override without creating another class.

```python
from minions.gemini import GovernanceMinion


minion = GovernanceMinion(
    model='gemini-2.5-flash',
    instructions='Evaluate the proposed AI system against the governance requirements.',
    name='AI Governance Reviewer',
)

print( minion.display_name )
print( minion.name )
```

Gemini normalizes the provider-facing `name` to an ADK-compatible identifier while preserving the
human-readable value in `display_name`.

## Use Configured Model Lists

`minions.config` exports provider-specific model lists for application controls and validation.

```python
from minions.config import (
    CLAUDE_MODELS,
    GEMINI_MODELS,
    GPT_MODELS,
    GROK_MODELS,
    MISTRAL_MODELS,
)


models_by_provider = {
    'OpenAI': GPT_MODELS,
    'Gemini': GEMINI_MODELS,
    'Grok': GROK_MODELS,
    'Claude': CLAUDE_MODELS,
    'Mistral': MISTRAL_MODELS,
}

for provider, models in models_by_provider.items( ):
    print( provider, models )
```

Model availability changes independently of Minions. Confirm that the configured provider account
can access the selected model.

## Validation and Failure Conditions

Minions validates configuration before provider execution.

| Condition | Result |
|---|---|
| Empty `model`, `instructions`, `name`, or `prompt` | `ValueError` |
| `max_turns < 1` | `ValueError` |
| Claude `max_tokens < 1` | `ValueError` |
| Missing Grok, Claude, or Mistral API key | `ValueError` |
| Duplicate local function names | `ValueError` |
| Grok or Mistral schema/function mismatch | `ValueError` |
| Local async function used with synchronous Grok or Mistral execution | `TypeError` |
| Workflow exceeds `max_turns` | `RuntimeError` |

Catch provider SDK errors at the application boundary. Preserve the original provider exception so
authentication, rate-limit, schema, and service errors remain diagnosable.

```python
from minions.gpt import DataMinion


try:
    minion = DataMinion(
        model='gpt-5.6-sol',
        instructions='Analyze the supplied data.',
        max_turns=10,
    )
    result = minion.run( 'Analyze the supplied records.' )
    print( result.final_output )
except ValueError as error:
    print( f'Invalid Minion configuration: {error}' )
except Exception as error:
    print( f'Provider execution failed: {error}' )
```

## Provider Boundary Rules

- Import the Minion and tools from matching provider pathways.
- Do not pass an OpenAI tool to Gemini, Grok, Claude, or Mistral.
- Use `functions` only for client-executed Grok and Mistral function schemas.
- Do not create local callables for provider-hosted Grok or Mistral tools.
- Keep tools optional when the workflow does not require external data or actions.
- Use the returned provider-native object instead of converting every provider to a common result
  type.
- Set `max_turns` according to the expected tool loop and operational limits.
