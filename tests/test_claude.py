'''Claude Minion execution tests.'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import beta_tool

from minions.claude import DataMinion


@beta_tool
def sample_tool( value: str ) -> str:
    '''Return a sample result.'''
    return value


@patch( 'minions.claude.AsyncAnthropic' )
@patch( 'minions.claude.Anthropic' )
def test_run_uses_native_tool_runner( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify Claude delegates the complete synchronous loop to Anthropic.'''
    expected = Mock( )
    runner = client_type.return_value.beta.messages.tool_runner.return_value
    runner.until_done.return_value = expected
    minion = DataMinion(
        model='claude-test', instructions='Analyze data.', tools=[ sample_tool ],
        api_key='test-key'
    )
    assert minion.run( 'Inspect this.' ) is expected


@pytest.mark.asyncio
@patch( 'minions.claude.AsyncAnthropic' )
@patch( 'minions.claude.Anthropic' )
async def test_run_async_uses_native_tool_runner( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify Claude delegates the complete asynchronous loop to Anthropic.'''
    expected = Mock( )
    runner = async_client_type.return_value.beta.messages.tool_runner.return_value
    runner.until_done = AsyncMock( return_value=expected )
    minion = DataMinion(
        model='claude-test', instructions='Analyze data.', tools=[ sample_tool ],
        api_key='test-key'
    )
    assert await minion.run_async( 'Inspect this.' ) is expected


@patch( 'minions.claude.AsyncAnthropic' )
@patch( 'minions.claude.Anthropic' )
def test_stream_enables_native_streaming_tool_runner( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify Claude streaming remains provider-managed across tool calls.'''
    minion = DataMinion(
        model='claude-test', instructions='Analyze data.', api_key='test-key'
    )
    result = minion.stream( 'Inspect this.' )
    assert result is async_client_type.return_value.beta.messages.tool_runner.return_value
    assert async_client_type.return_value.beta.messages.tool_runner.call_args.kwargs[ 'stream' ]
