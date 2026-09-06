'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                test_claude.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="test_claude.py" company="Terry D. Eppler">

         test_claude.py
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
        Claude Minion execution tests.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import beta_tool
from anthropic.types.beta import BetaToolUnionParam

from minions.claude import (
    BetaCodeExecutionTool20260521Param,
    BetaWebFetchTool20260318Param,
    BetaWebSearchTool20260318Param,
    DataMinion,
)


@beta_tool
def sample_tool( value: str ) -> str:
    '''Return a sample result.'''
    return value


@patch( 'minions.claude.AsyncAnthropic' )
@patch( 'minions.claude.Anthropic' )
def test_claude_server_tools_are_retained( client_type: Mock,
        async_client_type: Mock ) -> None:
    '''Verify Anthropic server tools remain unchanged for sync and async runners.'''
    tools: list[ BetaToolUnionParam ] = [
        BetaWebSearchTool20260318Param(
            type='web_search_20260318', name='web_search'
        ),
        BetaWebFetchTool20260318Param(
            type='web_fetch_20260318', name='web_fetch'
        ),
        BetaCodeExecutionTool20260521Param(
            type='code_execution_20260521', name='code_execution'
        ),
    ]
    minion = DataMinion(
        model='claude-test', instructions='Analyze data.', tools=tools,
        api_key='test-key'
    )

    assert minion.tools == tools
    assert minion.async_tools == tools

    minion.run( 'Inspect this.' )
    assert client_type.return_value.beta.messages.tool_runner.call_args.kwargs[ 'tools' ] == tools

    minion.stream( 'Inspect this.' )
    assert async_client_type.return_value.beta.messages.tool_runner.call_args.kwargs[ 'tools' ] \
        == tools


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
