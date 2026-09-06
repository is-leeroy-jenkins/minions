###### minions

<p align="center">
  <img src="resources/images/minions-project.png" alt="Minions" width="900">
</p>

<p align="center">
  <a href="#supported-providers">Providers</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#concrete-minions">Minions</a> ·
  <a href="#native-provider-tools">Native Tools</a> ·
  <a href="[user-guide.md](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/user-guide.md)">User Guide</a> ·
  <a href="#development">Development</a>
</p>

___

Provider-native AI agents organized as reusable workflow-specific `Minion` classes.

![](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/images/minions-workflow.png)

___

Minions keeps each workflow inside one provider pathway. Tool schemas and tool functions are never
translated or mixed between providers.

## Supported Providers

| Module            | Provider SDK                 | `Minion` implementation                  | Execution methods            |
|-------------------|------------------------------|------------------------------------------|------------------------------|
| `minions.gpt`     | OpenAI Agents SDK            | Inherits `agents.Agent`                  | `run`, `run_async`, `stream` |
| `minions.gemini`  | Google Agent Development Kit | Inherits `google.adk.Agent`              | `run`, `run_async`, `stream` |
| `minions.grok`    | xAI SDK                      | Wraps synchronous and asynchronous chats | `run`, `run_async`, `stream` |
| `minions.claude`  | Anthropic SDK                | Wraps native tool runners                | `run`, `run_async`, `stream` |
| `minions.mistral` | Mistral SDK                  | Wraps a native remote agent              | `run`, `run_async`, `stream` |

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install git+https://github.com/is-leeroy-jenkins/minions.git
```

Editable installation for development:

```powershell
git clone https://github.com/is-leeroy-jenkins/minions.git
Set-Location minions
python -m pip install -e .
```

## Authentication

Set only the key required by the selected provider.

| Provider | Environment variable |
|----------|----------------------|
| OpenAI   | `OPENAI_API_KEY`     |
| Gemini   | `GOOGLE_API_KEY`     |
| Grok     | `XAI_API_KEY`        |
| Claude   | `ANTHROPIC_API_KEY`  |
| Mistral  | `MISTRAL_API_KEY`    |



![](https://github.com/is-leeroy-jenkins/minions/blob/main/resources/images/minions-architecture.png)

___

## Concrete Minions

The same concrete class family is exported from every provider module.

| Guro workflow category                   | Minion class            |
|------------------------------------------|-------------------------|
| Research / Academic                      | `ResearchMinion`        |
| Writing / Administrative                 | `WritingMinion`         |
| Compliance / Legal / Budget              | `ComplianceMinion`      |
| Business / Finance / Marketing           | `BusinessMinion`        |
| Software Engineering / Software Engineer | `CodingMinion`          |
| Data Analytics                           | `DataMinion`            |
| Data Governance                          | `GovernanceMinion`      |
| Instruction / Training / Planning        | `PlanningMinion`        |
| Image Generation                         | `ImageGenerationMinion` |
| Image Analysis                           | `ImageAnalysisMinion`   |
| Image Editing                            | `ImageEditingMinion`    |
| Translation API                          | `TranslationMinion`     |
| Transcription API                        | `TranscriptionMinion`   |
| Speech API                               | `SpeechMinion`          |



## Quick Start

### OpenAI

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

### Gemini

```python
from fonky.gemini import tools
from guro import instructions
from minions.gemini import ResearchMinion


minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions=instructions.get( 'DEEP_RESEARCH_AGENT' ),
    tools=[ tools.fetch_wikipedia ],
)
result = minion.run( 'Research the requested topic.' )
```

### Grok

Grok requires matching provider schemas and local callables only for client-executed function
tools. xAI-hosted tools do not require entries in `functions`.

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

### Claude

```python
from fonky.claude import tools
from guro import instructions
from minions.claude import ComplianceMinion


minion = ComplianceMinion(
    model='claude-sonnet-4-6',
    instructions=instructions.get( 'COMPLIANCE_ANALYST' ),
    tools=[ tools.load_pdf, tools.fetch_web_page ],
)
result = minion.run( 'Evaluate the supplied material.' )
```

### Mistral

Mistral requires matching provider schemas and local callables for local function tools.
Provider-hosted tools do not require a local callable.

```python
from fonky.mistral import tools
from guro import instructions
from minions.mistral import CodingMinion


minion = CodingMinion(
    model='mistral-medium-latest',
    instructions=instructions.get( 'SENIOR_ENGINEER' ),
    tools=[ tools.github_tool ],
    functions=[ tools.load_github ],
)
result = minion.run( 'Review the repository.' )
```

## Tool-Free Execution

Tools are optional for every provider.

```python
from guro import instructions
from minions.gpt import WritingMinion


minion = WritingMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'TECHNICAL_WRITER' ),
)
result = minion.run( 'Draft the requested documentation.' )
```

## Constructor Reference

### Common arguments

| Argument | Type | Default | Description |
|---|---|---:|---|
| `model` | `str` | Required | Provider model identifier |
| `instructions` | `str` | Required | System instructions, including Guro instruction text |
| `tools` | Provider-specific sequence or `None` | `None` | Optional provider-native tools |
| `max_turns` | `int` | `10` | Maximum model turns for one execution |
| `name` | `str` or `None` | `None` | Overrides the concrete class display name |

### Provider-specific arguments

| Provider | Argument | Type | Default | Description |
|---|---|---|---:|---|
| Grok | `functions` | `Sequence[ToolFunction]` or `None` | `None` | Local callables matching Grok tool schemas |
| Grok | `api_key` | `str` or `None` | `None` | Overrides `XAI_API_KEY` |
| Claude | `max_tokens` | `int` | `4096` | Maximum generated tokens |
| Claude | `api_key` | `str` or `None` | `None` | Overrides `ANTHROPIC_API_KEY` |
| Mistral | `functions` | `Sequence[ToolFunction]` or `None` | `None` | Local callables matching function-tool schemas |
| Mistral | `api_key` | `str` or `None` | `None` | Overrides `MISTRAL_API_KEY` |

## Execution Reference

| Provider | `run` return | `run_async` return | `stream` return |
|---|---|---|---|
| OpenAI | `RunResult` | `RunResult` | `RunResultStreaming` |
| Gemini | Final `Event` | Final `Event` | `AsyncIterator[Event]` |
| Grok | `Response` | `Response` | `AsyncIterator[tuple[Response, Chunk]]` |
| Claude | `BetaMessage` | `BetaMessage` | `BetaAsyncStreamingToolRunner[object]` |
| Mistral | `ChatCompletionResponse` | `ChatCompletionResponse` | `Iterator[CompletionEvent]` |

Asynchronous execution:

```python
result = await minion.run_async( 'Complete the workflow.' )
```

Streaming execution:

```python
stream = minion.stream( 'Complete the workflow.' )
```

OpenAI and Claude return provider-managed streaming objects. Gemini and Grok return asynchronous
iterators. Mistral returns a synchronous iterator. Each pathway continues its tool loop through the
final provider response.

## Tool Contracts

| Provider | Accepted tool contract |
|---|---|
| OpenAI | OpenAI Agents SDK `Tool` objects |
| Gemini | Callable, `BaseTool`, or `BaseToolset` |
| Grok | xAI `chat_pb2.Tool`; local function schemas require identically named callables |
| Claude | `ClaudeTool`: Anthropic `BetaFunctionTool` or `BetaToolUnionParam` |
| Mistral | `CreateAgentRequestTool` schemas; local function tools require identically named callables |

Grok and Mistral reject duplicate, missing, or extra local function names before execution.
Provider-hosted tools execute on the provider and therefore do not require local callables.

## Native Provider Tools

| Provider | Native tools exported by the provider module |
|---|---|
| OpenAI | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool` |
| Gemini | `google_search`, `url_context`, `VertexAiSearchTool` |
| Grok | `web_search`, `code_execution`, `collections_search`, `image_generation` |
| Claude | Web search, web fetch, and code execution through `BetaToolUnionParam` definitions |
| Mistral | `WebSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `DocumentLibraryTool` |

Native tools remain optional and provider-specific. Do not pass a native tool from one provider to
another provider's Minion.

### OpenAI native tools

```python
from minions.gpt import FileSearchTool, ResearchMinion, WebSearchTool


minion = ResearchMinion(
    model='gpt-5.6-sol',
    instructions='Research the question using current and indexed sources.',
    tools=[
        WebSearchTool( ),
        FileSearchTool( vector_store_ids=[ 'vs_...' ] ),
    ],
)
```

`CodeInterpreterTool` and `ImageGenerationTool` accept their native OpenAI tool configuration
objects. Minions passes those configurations to the OpenAI Agents SDK unchanged.

### Gemini native tools

```python
from minions.gemini import ResearchMinion, VertexAiSearchTool


minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions='Research the configured enterprise data store.',
    tools=[
        VertexAiSearchTool(
            data_store_id=(
                'projects/project/locations/global/collections/default_collection/'
                'dataStores/store'
            ),
        ),
    ],
)
```

`google_search` and `url_context` are native ADK tool objects and can be imported directly from
`minions.gemini`. Supported tool combinations depend on the selected Gemini model and ADK rules.

### Grok native tools

```python
from minions.grok import DataMinion, code_execution, collections_search, web_search


minion = DataMinion(
    model='grok-4.5',
    instructions='Research and analyze the requested subject.',
    tools=[
        web_search( ),
        code_execution( ),
        collections_search( collection_ids=[ 'collection-id' ] ),
    ],
)
```

The `image_generation` constructor is also exported from `minions.grok`. xAI executes these tools
on its servers; `functions` remains reserved for client-executed function schemas.

### Claude native tools

```python
from minions.claude import ResearchMinion


minion = ResearchMinion(
    model='claude-sonnet-4-6',
    instructions='Research and analyze the requested subject.',
    tools=[
        { 'type': 'web_search_20260318', 'name': 'web_search' },
        { 'type': 'web_fetch_20260318', 'name': 'web_fetch' },
        { 'type': 'code_execution_20260521', 'name': 'code_execution' },
    ],
)
```

Anthropic executes these server tools. Local tools decorated with `anthropic.beta_tool` can appear
in the same `tools` sequence and are executed by Anthropic's native tool runner.

### Mistral native tools

```python
from minions.mistral import (
    CodeInterpreterTool,
    DataMinion,
    DocumentLibraryTool,
    WebSearchTool,
)


minion = DataMinion(
    model='mistral-medium-latest',
    instructions='Research and analyze the requested subject.',
    tools=[
        WebSearchTool( ),
        CodeInterpreterTool( ),
        DocumentLibraryTool( library_ids=[ 'library-id' ] ),
    ],
)
```

`ImageGenerationTool` is also exported from `minions.mistral`. Mistral executes native tools;
`functions` remains reserved for client-executed function schemas.

## Category Templates

Concrete Minions are subclass templates. Guro prompts plug into a category through the
`instructions` argument without a factory or registry.

```python
from guro import instructions
from minions.gpt import GovernanceMinion


governor = GovernanceMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'AI_GOVENANCE_AGENT' ),
)
```

Provider-specific customization uses ordinary inheritance:

```python
from minions.gpt import ComplianceMinion


class BudgetMinion( ComplianceMinion ):
    """OpenAI Minion specialized for federal budget compliance."""

    minion_name: str = 'Budget Minion'
```

## Development

```powershell
python -m pip install -e .
python -m pip install pytest pytest-asyncio build
python -m pytest
python -m build
```

Documentation:

```powershell
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

Tests mock provider network boundaries and validate provider inheritance, optional tools, concrete
class exports, synchronous execution, asynchronous execution, streaming, and local tool loops.

## Project Structure

```text
minions/
├── __init__.py
├── config.py
├── gpt.py
├── gemini.py
├── grok.py
├── claude.py
├── mistral.py
└── tests/
```
<p align="center">
  <a href="https://github.com/is-leeroy-jenkins/minions/actions/workflows/tests.yml"><img src="https://github.com/is-leeroy-jenkins/minions/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/version-0.2.0-6f42c1" alt="Version 0.2.0">
</p>
