'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                test_mistral.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="test_mistral.py" company="Terry D. Eppler">

         test_mistral.py
         Copyright © 2026 Terry D. Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
     PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
     CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
     OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

     You can contact me at: terryeppler@gmail.com or eppler.terry@epa.gov

    </copyright>
    <summary>
        Mistral Minion execution tests.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from mistralai.client.models import (
    AssistantMessage,
    ChatCompletionResponse,
    CompletionChunk,
    CompletionEvent,
    CodeInterpreterTool,
    DocumentLibraryTool,
    ImageGenerationTool,
    ToolCall,
    WebSearchTool,
)

from minions.mistral import DataMinion


def sample_tool( value: str ) -> dict[ str, str ]:
    '''Return a sample tool result.'''
    return { 'value': value }


sample_schema = {
    'type': 'function',
    'function': {
        'name': 'sample_tool',
        'description': 'Return a sample result.',
        'parameters': {
            'type': 'object',
            'properties': { 'value': { 'type': 'string' } },
            'required': [ 'value' ],
        },
    },
}


def response( tool_calls: list[ ToolCall ] | None=None ) -> ChatCompletionResponse:
    '''Create a provider-native completion response.'''
    return ChatCompletionResponse.model_validate( {
        'id': 'response-1',
        'object': 'chat.completion',
        'model': 'mistral-test',
        'created': 0,
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        },
        'choices': [ {
            'index': 0,
            'message': AssistantMessage( content='done', tool_calls=tool_calls ),
            'finish_reason': 'tool_calls' if tool_calls else 'stop',
        } ],
    } )


def requested_call( ) -> ToolCall:
    '''Create a provider-native requested tool call.'''
    return ToolCall.model_validate( {
        'id': 'call-1',
        'type': 'function',
        'function': { 'name': 'sample_tool', 'arguments': '{"value":"records"}' },
    } )


@patch( 'minions.mistral.Mistral' )
def test_mistral_native_tools_do_not_require_local_functions(
        client_type: Mock ) -> None:
    '''Verify every supported Mistral-hosted tool bypasses local function validation.'''
    tools = [
        WebSearchTool( ),
        CodeInterpreterTool( ),
        ImageGenerationTool( ),
        DocumentLibraryTool( library_ids=[ 'library-id' ] ),
    ]
    minion = DataMinion(
        model='mistral-test', instructions='Analyze data.', tools=tools,
        api_key='test-key'
    )

    assert minion.tools == tools
    assert minion.functions == { }
    assert client_type.return_value.beta.agents.create.call_args.kwargs[ 'tools' ] == tools


@patch( 'minions.mistral.Mistral' )
def test_run_executes_tool_to_completion( client_type: Mock ) -> None:
    '''Verify synchronous Mistral execution resumes after local tools.'''
    client = client_type.return_value
    client.beta.agents.create.return_value.id = 'agent-1'
    final = response( )
    client.agents.complete.side_effect = [ response( [ requested_call( ) ] ), final ]
    minion = DataMinion(
        model='mistral-test', instructions='Analyze data.', tools=[ sample_schema ],
        functions=[ sample_tool ], api_key='test-key'
    )
    assert minion.run( 'Inspect this.' ) is final
    assert client.agents.complete.call_count == 2


@pytest.mark.asyncio
@patch( 'minions.mistral.Mistral' )
async def test_run_async_executes_tool_to_completion( client_type: Mock ) -> None:
    '''Verify asynchronous Mistral execution resumes after local tools.'''
    client = client_type.return_value
    client.beta.agents.create.return_value.id = 'agent-1'
    final = response( )
    client.agents.complete_async = AsyncMock(
        side_effect=[ response( [ requested_call( ) ] ), final ]
    )
    minion = DataMinion(
        model='mistral-test', instructions='Analyze data.', tools=[ sample_schema ],
        functions=[ sample_tool ], api_key='test-key'
    )
    assert await minion.run_async( 'Inspect this.' ) is final


def chunk( finish_reason: str, tool_calls: list[ dict[ str, object ] ] | None=None
        ) -> CompletionEvent:
    '''Create a provider-native Mistral streaming event.'''
    return CompletionEvent( data=CompletionChunk.model_validate( {
        'id': 'response-1',
        'model': 'mistral-test',
        'choices': [ {
            'index': 0,
            'delta': { 'role': 'assistant', 'content': '', 'tool_calls': tool_calls },
            'finish_reason': finish_reason,
        } ],
    } ) )


@patch( 'minions.mistral.Mistral' )
def test_stream_executes_tool_and_resumes( client_type: Mock ) -> None:
    '''Verify Mistral streaming owns the complete tool lifecycle.'''
    client = client_type.return_value
    client.beta.agents.create.return_value.id = 'agent-1'
    requested = chunk( 'tool_calls', [ {
        'index': 0,
        'id': 'call-1',
        'type': 'function',
        'function': { 'name': 'sample_tool', 'arguments': '{"value":"records"}' },
    } ] )
    final = chunk( 'stop' )
    client.agents.stream.side_effect = [ iter( [ requested ] ), iter( [ final ] ) ]
    minion = DataMinion(
        model='mistral-test', instructions='Analyze data.', tools=[ sample_schema ],
        functions=[ sample_tool ], api_key='test-key'
    )
    assert list( minion.stream( 'Inspect this.' ) ) == [ requested, final ]
    assert client.agents.stream.call_count == 2


@patch( 'minions.mistral.Mistral' )
def test_mismatched_mistral_tools_are_rejected( client_type: Mock ) -> None:
    '''Verify provider schema and callable sets must match exactly.'''
    with pytest.raises( ValueError, match='match exactly' ):
        DataMinion(
            model='mistral-test', instructions='Analyze data.', tools=[ sample_schema ],
            api_key='test-key'
        )


@patch( 'minions.mistral.Mistral' )
def test_provider_hosted_mistral_tools_do_not_require_functions(
        client_type: Mock ) -> None:
    '''Verify Mistral-hosted tools are not mistaken for local functions.'''
    minion = DataMinion(
        model='mistral-test',
        instructions='Research the question.',
        tools=[ { 'type': 'web_search' } ],
        api_key='test-key',
    )
    assert minion.functions == { }
