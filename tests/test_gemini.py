'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                test_gemini.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="test_gemini.py" company="Terry D. Eppler">

         test_gemini.py
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
        Gemini Minion execution tests.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from unittest.mock import Mock, patch

import pytest
from google.adk.agents.run_config import StreamingMode
from google.adk.events import Event

from minions.gemini import (
    DataMinion,
    VertexAiSearchTool,
    google_search,
    url_context,
)


def emit( event: Event ) -> Iterator[ Event ]:
    '''Yield one synchronous ADK event.'''
    yield event


async def emit_async( event: Event ) -> AsyncIterator[ Event ]:
    '''Yield one asynchronous ADK event.'''
    yield event


@pytest.mark.parametrize( 'native_tool', [
    google_search,
    url_context,
    VertexAiSearchTool(
        data_store_id=(
            'projects/project/locations/global/collections/default_collection/'
            'dataStores/store'
        )
    ),
] )
def test_gemini_native_tools_are_retained( native_tool: object ) -> None:
    '''Verify every supported Gemini-native tool reaches the ADK Agent unchanged.'''
    minion = DataMinion(
        model='gemini-test', instructions='Analyze data.', tools=[ native_tool ]
    )

    assert minion.tools == [ native_tool ]


@patch( 'minions.gemini.Runner' )
def test_run_returns_final_native_event( runner_type: Mock ) -> None:
    '''Verify synchronous ADK execution and automatic tool handling.'''
    event = Mock( spec=Event )
    event.is_final_response.return_value = True
    runner_type.return_value.run.return_value = emit( event )
    minion = DataMinion( model='gemini-test', instructions='Analyze data.' )
    assert minion.run( 'Inspect this.' ) is event


@pytest.mark.asyncio
@patch( 'minions.gemini.Runner' )
async def test_stream_uses_sse_and_returns_all_events( runner_type: Mock ) -> None:
    '''Verify streaming requests SSE and retains the final native event.'''
    event = Mock( spec=Event )
    event.is_final_response.return_value = True
    runner_type.return_value.run_async.return_value = emit_async( event )
    minion = DataMinion( model='gemini-test', instructions='Analyze data.' )

    assert [ item async for item in minion.stream( 'Inspect this.' ) ] == [ event ]
    config = runner_type.return_value.run_async.call_args.kwargs[ 'run_config' ]
    assert config.streaming_mode is StreamingMode.SSE
