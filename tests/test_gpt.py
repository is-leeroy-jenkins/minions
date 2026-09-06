'''OpenAI Minion execution tests.'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minions.gpt import DataMinion


@patch( 'minions.gpt.Runner.run_sync' )
def test_run_returns_native_result( run_sync: Mock ) -> None:
    '''Verify synchronous execution uses the Minion itself as the native Agent.'''
    expected = Mock( )
    run_sync.return_value = expected
    minion = DataMinion( model='gpt-test', instructions='Analyze data.' )

    result = minion.run( 'Inspect this.' )

    run_sync.assert_called_once_with( minion, 'Inspect this.', max_turns=10 )
    assert result is expected


@pytest.mark.asyncio
@patch( 'minions.gpt.Runner.run', new_callable=AsyncMock )
async def test_run_async_returns_native_result( run: AsyncMock ) -> None:
    '''Verify asynchronous execution returns the native result.'''
    expected = Mock( )
    run.return_value = expected
    minion = DataMinion( model='gpt-test', instructions='Analyze data.' )
    assert await minion.run_async( 'Inspect this.' ) is expected


@patch( 'minions.gpt.Runner.run_streamed' )
def test_stream_returns_native_tool_managed_stream( run_streamed: Mock ) -> None:
    '''Verify streaming delegates its complete tool lifecycle to OpenAI.'''
    expected = Mock( )
    run_streamed.return_value = expected
    minion = DataMinion( model='gpt-test', instructions='Analyze data.' )
    assert minion.stream( 'Inspect this.' ) is expected
