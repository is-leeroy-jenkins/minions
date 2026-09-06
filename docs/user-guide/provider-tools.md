# Provider Tools

Native tools are optional and remain owned by their provider SDK. Minions re-exports the commonly
used hosted tools so a workflow can be configured from one module.

## Tool matrix

| Provider | Hosted tools exposed by Minions |
| --- | --- |
| OpenAI | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool` |
| Gemini | `google_search`, `url_context`, `VertexAiSearchTool` |
| Grok | `web_search`, `code_execution`, `collections_search`, `image_generation` |
| Claude | Typed web search, web fetch, and code execution definitions |
| Mistral | `WebSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `DocumentLibraryTool` |

## OpenAI: web and indexed file search

```python
from minions.gpt import FileSearchTool, ResearchMinion, WebSearchTool


minion = ResearchMinion(
    model='gpt-5.6-sol',
    instructions='Use current and indexed sources. Cite the evidence for each finding.',
    tools=[
        WebSearchTool( ),
        FileSearchTool( vector_store_ids=[ 'vs_1234567890' ] ),
    ],
)

result = minion.run( 'Compare the current policy with the indexed requirements.' )
print( result.final_output )
```

`CodeInterpreterTool` and `ImageGenerationTool` are passed to the OpenAI Agents SDK unchanged.

## Gemini: Google Search

```python
from minions.gemini import ResearchMinion, google_search


minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions='Use Google Search and attribute the supporting sources.',
    tools=[ google_search ],
)

event = minion.run( 'Summarize current developments in agent SDKs.' )
for part in event.content.parts:
    if part.text:
        print( part.text )
```

Use `url_context` to supply page context or `VertexAiSearchTool` for enterprise retrieval:

```python
from minions.gemini import ResearchMinion, VertexAiSearchTool


data_store = (
    'projects/example/locations/global/collections/default_collection/'
    'dataStores/policy-library'
)

minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions='Search the enterprise policy library and cite supporting documents.',
    tools=[ VertexAiSearchTool( data_store_id=data_store ) ],
)
```

## Grok: hosted and local tools

xAI-hosted tools need no local callable:

```python
from minions.grok import DataMinion, code_execution, web_search


minion = DataMinion(
    model='grok-4.5',
    instructions='Research current evidence and verify calculations with code.',
    tools=[ web_search( ), code_execution( ) ],
)
```

A client-executed function requires a schema in `tools` and an identically named callable in
`functions`:

```python
from xai_sdk.chat import tool
from minions.grok import DataMinion


def lookup_account( account_id: str ) -> dict[ str, str ]:
    """Return an account record."""
    return { 'account_id': account_id, 'status': 'active' }


schema = tool(
    name='lookup_account',
    description='Return one account record.',
    parameters={
        'type': 'object',
        'properties': { 'account_id': { 'type': 'string' } },
        'required': [ 'account_id' ],
    },
)

minion = DataMinion(
    model='grok-4.5',
    instructions='Use the account tool for account-status questions.',
    tools=[ schema ],
    functions=[ lookup_account ],
)
```

## Claude: server and local tools

```python
from minions.claude import (
    BetaCodeExecutionTool20260521Param,
    BetaWebFetchTool20260318Param,
    BetaWebSearchTool20260318Param,
    ResearchMinion,
)


minion = ResearchMinion(
    model='claude-sonnet-4-6',
    instructions='Research, inspect relevant pages, and verify calculations.',
    tools=[
        BetaWebSearchTool20260318Param(
            type='web_search_20260318', name='web_search'
        ),
        BetaWebFetchTool20260318Param(
            type='web_fetch_20260318', name='web_fetch'
        ),
        BetaCodeExecutionTool20260521Param(
            type='code_execution_20260521', name='code_execution'
        ),
    ],
)
```

Local functions decorated with `anthropic.beta_tool` can appear in the same sequence. The native
Anthropic tool runner executes the full lifecycle.

## Mistral: hosted and local tools

```python
from minions.mistral import (
    CodeInterpreterTool,
    DocumentLibraryTool,
    ResearchMinion,
    WebSearchTool,
)


minion = ResearchMinion(
    model='mistral-medium-latest',
    instructions='Use web evidence, the document library, and code where appropriate.',
    tools=[
        WebSearchTool( ),
        CodeInterpreterTool( ),
        DocumentLibraryTool( library_ids=[ 'library-id' ] ),
    ],
)
```

As with Grok, a local function-tool schema requires an identically named callable in `functions`.
Duplicate, missing, or extra function names are rejected during construction.
