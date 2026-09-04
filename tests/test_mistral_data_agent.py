'''Tests for the Mistral AI DataAgent implementation.'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minions.mistral import DataAgent


def sample_tool( value: str ) -> dict[ str, str ]:
    '''Return a provider-compatible test result.'''
    return { 'value': value }


def create_agent( ) -> DataAgent:
    '''Create a DataAgent configured with a Mistral-compatible test tool.'''
    return DataAgent(
        name='Data Agent',
        model='mistral-small-latest',
        instructions='Analyze the supplied data.',
        tools=[ sample_tool ],
        max_turns=7,
        api_key='test-key',
    )


@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
def test_constructor_creates_mistral_agent( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify that stored members are used to create the Mistral agent.'''
    schema = Mock( name='schema' )
    mock_create_tool.return_value = schema

    agent = create_agent( )

    mock_mistral.assert_called_once_with( api_key=agent.api_key )
    mock_create_tool.assert_called_once_with( sample_tool )
    agent.client.beta.agents.create.assert_called_once_with(
        name=agent.name,
        model=agent.model,
        instructions=agent.instructions,
        tools=[ schema ],
    )
    assert agent.provider is agent.client.beta.agents.create.return_value


@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
def test_run_executes_tool_and_returns_final_response( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify synchronous model turns and local tool execution.'''
    tool_call = Mock( id='call-1' )
    tool_call.function.name = 'sample_tool'
    tool_call.function.arguments = '{"value": "records"}'
    requested = Mock( )
    requested.choices = [ Mock( ) ]
    requested.choices[ 0 ].message.tool_calls = [ tool_call ]
    final = Mock( )
    final.choices = [ Mock( ) ]
    final.choices[ 0 ].message.tool_calls = [ ]
    agent = create_agent( )
    agent.client.agents.complete.side_effect = [ requested, final ]

    result = agent.run( 'Inspect the dataset.' )

    assert agent.client.agents.complete.call_count == 2
    assert result is final
    assert agent.result is final
    messages = agent.client.agents.complete.call_args.kwargs[ 'messages' ]
    assert messages[ -1 ] == {
        'role': 'tool',
        'name': 'sample_tool',
        'content': '{"value": "records"}',
        'tool_call_id': 'call-1',
    }


@pytest.mark.asyncio
@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
async def test_run_async_returns_final_response( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify asynchronous execution and provider-native result retention.'''
    final = Mock( )
    final.choices = [ Mock( ) ]
    final.choices[ 0 ].message.tool_calls = [ ]
    agent = create_agent( )
    agent.client.agents.complete_async = AsyncMock( return_value=final )

    result = await agent.run_async( 'Inspect the dataset.' )

    agent.client.agents.complete_async.assert_awaited_once_with(
        agent_id=agent.provider.id,
        messages=[ { 'role': 'user', 'content': agent.prompt } ],
    )
    assert result is final
    assert agent.result is final


@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
def test_stream_returns_provider_stream( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify streamed execution and provider-native stream retention.'''
    expected = Mock( name='stream' )
    agent = create_agent( )
    agent.client.agents.stream.return_value = expected

    result = agent.stream( 'Inspect the dataset.' )

    agent.client.agents.stream.assert_called_once_with(
        agent_id=agent.provider.id,
        messages=[ { 'role': 'user', 'content': agent.prompt } ],
    )
    assert result is expected
    assert agent.result is expected


@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
def test_constructor_rejects_noncallable_tools( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify that Mistral DataAgent accepts only executable callable tools.'''
    with pytest.raises( ValueError ):
        DataAgent(
            name='Data Agent',
            model='mistral-small-latest',
            instructions='Analyze the supplied data.',
            tools=[ { 'type': 'function' } ],
            api_key='test-key',
        )


@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
def test_execution_rejects_empty_prompt( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify that synchronous and streaming execution reject an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        agent.run( '' )

    with pytest.raises( ValueError ):
        agent.stream( '' )


@pytest.mark.asyncio
@patch( 'minions.mistral.agents.create_tool_call' )
@patch( 'minions.mistral.agents.Mistral' )
async def test_async_execution_rejects_empty_prompt( mock_mistral: Mock,
        mock_create_tool: Mock ) -> None:
    '''Verify that asynchronous execution rejects an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        await agent.run_async( '' )
