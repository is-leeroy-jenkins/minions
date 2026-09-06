'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                test_grok.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="test_grok.py" company="Terry D. Eppler">

         test_grok.py
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
        Grok Minion execution tests.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from xai_sdk.chat import tool

from minions.grok import DataMinion


def sample_tool( value: str ) -> dict[ str, str ]:
    '''Return a sample tool result.'''
    return { 'value': value }


sample_schema = tool(
    name='sample_tool',
    description='Return a sample tool result.',
    parameters={
        'type': 'object',
        'properties': { 'value': { 'type': 'string' } },
        'required': [ 'value' ],
    },
)


def tool_call( ) -> SimpleNamespace:
    '''Create a representative xAI tool call.'''
    function = SimpleNamespace( name='sample_tool', arguments='{"value":"records"}' )
    return SimpleNamespace( id='call-1', function=function )


async def emit( *events: tuple[ object, object ] ) -> AsyncIterator[ tuple[ object, object ] ]:
    '''Yield provider stream pairs.'''
    for event in events:
        yield event


@patch( 'minions.grok.AsyncClient' )
@patch( 'minions.grok.Client' )
def test_run_executes_tool_to_completion( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify synchronous Grok tool execution resumes the model.'''
    requested = SimpleNamespace( tool_calls=[ tool_call( ) ] )
    final = SimpleNamespace( tool_calls=[ ] )
    chat = client_type.return_value.chat.create.return_value
    chat.sample.side_effect = [ requested, final ]
    minion = DataMinion(
        model='grok-test', instructions='Analyze data.', tools=[ sample_schema ],
        functions=[ sample_tool ], api_key='test-key'
    )
    assert minion.run( 'Inspect this.' ) is final
    assert chat.sample.call_count == 2


@pytest.mark.asyncio
@patch( 'minions.grok.AsyncClient' )
@patch( 'minions.grok.Client' )
async def test_stream_executes_tool_and_resumes( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify streamed Grok execution is complete rather than caller-controlled.'''
    requested = SimpleNamespace( tool_calls=[ tool_call( ) ] )
    final = SimpleNamespace( tool_calls=[ ] )
    chat = async_client_type.return_value.chat.create.return_value
    chat.stream.side_effect = [
        emit( ( requested, 'tool-chunk' ) ),
        emit( ( final, 'final-chunk' ) ),
    ]
    minion = DataMinion(
        model='grok-test', instructions='Analyze data.', tools=[ sample_schema ],
        functions=[ sample_tool ], api_key='test-key'
    )
    events = [ event async for event in minion.stream( 'Inspect this.' ) ]
    assert events == [ ( requested, 'tool-chunk' ), ( final, 'final-chunk' ) ]
    assert chat.stream.call_count == 2


@patch( 'minions.grok.AsyncClient' )
@patch( 'minions.grok.Client' )
def test_mismatched_grok_tools_are_rejected( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify provider schema and callable sets must match exactly.'''
    with pytest.raises( ValueError, match='match exactly' ):
        DataMinion(
            model='grok-test', instructions='Analyze data.', tools=[ sample_schema ],
            api_key='test-key'
        )
