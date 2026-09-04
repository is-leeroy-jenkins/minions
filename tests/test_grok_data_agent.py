'''Tests for the xAI Grok DataAgent implementation.'''
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from minions.grok import DataAgent


def sample_tool( value: str ) -> dict[ str, str ]:
    '''Return a provider-compatible test result.'''
    return { 'value': value }


def create_schema( name: str='sample_tool' ) -> Any:
    '''Create an xAI-compatible function schema.'''
    return SimpleNamespace( function=SimpleNamespace( name=name ) )


def create_tool_call( value: str='records' ) -> Any:
    '''Create a provider-native xAI tool call.'''
    function = SimpleNamespace( name='sample_tool', arguments=f'{{"value": "{value}"}}' )
    return SimpleNamespace( id='call-1', function=function )


def create_agent( ) -> DataAgent:
    '''Create a DataAgent configured with paired Grok schema and callable tools.'''
    return DataAgent(
        name='Data Agent',
        model='grok-test',
        instructions='Analyze the supplied data.',
        tools=[ create_schema( ) ],
        functions=[ sample_tool ],
        max_turns=7,
        api_key='test-key',
    )


async def emit( *events: Any ) -> AsyncIterator[ tuple[ Any, Any ] ]:
    '''Yield provider stream entries for asynchronous tests.'''
    for event in events:
        yield event


@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
def test_constructor_creates_xai_clients( mock_client: Mock,
        mock_async_client: Mock ) -> None:
    '''Verify that validated credentials create both xAI clients.'''
    agent = create_agent( )

    mock_client.assert_called_once_with( api_key=agent.api_key )
    mock_async_client.assert_called_once_with( api_key=agent.api_key )
    assert agent.provider is mock_client.return_value
    assert agent.functions == { 'sample_tool': sample_tool }


@patch( 'minions.grok.agents.tool_result' )
@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
def test_run_executes_tool_and_returns_final_response( mock_client: Mock,
        mock_async_client: Mock, mock_tool_result: Mock ) -> None:
    '''Verify synchronous Grok turns and correlated local tool execution.'''
    requested = SimpleNamespace( tool_calls=[ create_tool_call( ) ] )
    final = SimpleNamespace( tool_calls=[ ] )
    chat = mock_client.return_value.chat.create.return_value
    chat.sample.side_effect = [ requested, final ]
    agent = create_agent( )

    result = agent.run( 'Inspect the dataset.' )

    assert chat.sample.call_count == 2
    mock_tool_result.assert_called_once_with(
        '{"value": "records"}',
        tool_call_id='call-1',
    )
    chat.append.assert_any_call( mock_tool_result.return_value )
    assert result is final


@pytest.mark.asyncio
@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
async def test_run_async_returns_final_response( mock_client: Mock,
        mock_async_client: Mock ) -> None:
    '''Verify asynchronous Grok execution and provider-native result retention.'''
    final = SimpleNamespace( tool_calls=[ ] )
    chat = mock_async_client.return_value.chat.create.return_value
    chat.sample = AsyncMock( return_value=final )
    agent = create_agent( )

    result = await agent.run_async( 'Inspect the dataset.' )

    chat.sample.assert_awaited_once_with( )
    assert result is final
    assert agent.result is final


@pytest.mark.asyncio
@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
async def test_stream_completes_tool_loop( mock_client: Mock,
        mock_async_client: Mock ) -> None:
    '''Verify that Grok streaming continues until a tool-free final response.'''
    requested = SimpleNamespace( tool_calls=[ create_tool_call( ) ] )
    final = SimpleNamespace( tool_calls=[ ] )
    chat = mock_async_client.return_value.chat.create.return_value
    chat.stream.side_effect = [
        emit( ( requested, 'tool-chunk' ) ),
        emit( ( final, 'final-chunk' ) ),
    ]
    agent = create_agent( )

    events = [ event async for event in agent.stream( 'Inspect the dataset.' ) ]

    assert events == [ ( requested, 'tool-chunk' ), ( final, 'final-chunk' ) ]
    assert chat.stream.call_count == 2
    assert chat.append.call_count == 4


@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
def test_constructor_rejects_mismatched_tools( mock_client: Mock,
        mock_async_client: Mock ) -> None:
    '''Verify that xAI schemas and executable functions must match exactly.'''
    with pytest.raises( ValueError, match='must match exactly' ):
        DataAgent(
            name='Data Agent',
            model='grok-test',
            instructions='Analyze the supplied data.',
            tools=[ create_schema( 'different_tool' ) ],
            functions=[ sample_tool ],
            api_key='test-key',
        )


@patch( 'minions.grok.agents.AsyncClient' )
@patch( 'minions.grok.agents.Client' )
def test_execution_rejects_empty_prompt( mock_client: Mock,
        mock_async_client: Mock ) -> None:
    '''Verify that synchronous and streaming execution reject an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        agent.run( '' )

    with pytest.raises( ValueError ):
        agent.stream( '' )

