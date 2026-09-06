'''Gemini Minion execution tests.'''
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from unittest.mock import Mock, patch

import pytest
from google.adk.agents.run_config import StreamingMode
from google.adk.events import Event

from minions.gemini import DataMinion


def emit( event: Event ) -> Iterator[ Event ]:
    '''Yield one synchronous ADK event.'''
    yield event


async def emit_async( event: Event ) -> AsyncIterator[ Event ]:
    '''Yield one asynchronous ADK event.'''
    yield event


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
