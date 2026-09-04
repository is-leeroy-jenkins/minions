'''Tests for the Anthropic Claude DataAgent implementation.'''
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from minions.claude import DataAgent


class SampleTool:
    '''Provider-compatible Anthropic beta function tool used by tests.'''

    name = 'sample_tool'
    description = 'Return a provider-compatible test result.'
    input_schema = {
        'type': 'object',
        'properties': { 'value': { 'type': 'string' } },
        'required': [ 'value' ],
    }


    def to_dict( self ) -> dict[ str, Any ]:
        '''Return the Anthropic tool schema.'''
        return self.input_schema


    def call( self, **arguments: Any ) -> Any:
        '''Execute the provider tool.'''
        return self.func( **arguments )


    def func( self, value: str ) -> dict[ str, str ]:
        '''Return the supplied test value.'''
        return { 'value': value }


def create_agent( ) -> DataAgent:
    '''Create a DataAgent configured with an Anthropic beta tool.'''
    return DataAgent(
        name='Data Agent',
        model='claude-test',
        instructions='Analyze the supplied data.',
        tools=[ SampleTool( ) ],
        max_turns=7,
        max_tokens=2048,
        api_key='test-key',
    )


@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
def test_constructor_creates_anthropic_clients( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify client creation and schema-preserving asynchronous tool adaptation.'''
    agent = create_agent( )
    tool = agent.tools[ 0 ]

    mock_client.assert_called_once_with( api_key=agent.api_key )
    mock_async_client.assert_called_once_with( api_key=agent.api_key )
    arguments = mock_beta_async_tool.call_args.kwargs
    assert arguments[ 'name' ] == tool.name
    assert arguments[ 'description' ] == tool.description
    assert arguments[ 'input_schema' ] is tool.input_schema
    assert agent.provider is mock_client.return_value


@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
def test_run_returns_final_provider_message( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify synchronous execution through Anthropic's complete tool runner.'''
    expected = Mock( name='message' )
    runner = mock_client.return_value.beta.messages.tool_runner.return_value
    runner.until_done.return_value = expected
    agent = create_agent( )

    result = agent.run( 'Inspect the dataset.' )

    mock_client.return_value.beta.messages.tool_runner.assert_called_once_with(
        model=agent.model,
        max_tokens=agent.max_tokens,
        max_iterations=agent.max_turns,
        system=agent.instructions,
        tools=agent.tools,
        messages=[ { 'role': 'user', 'content': agent.prompt } ],
    )
    assert result is expected


@pytest.mark.asyncio
@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
async def test_run_async_returns_final_provider_message( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify asynchronous execution through Anthropic's complete tool runner.'''
    expected = Mock( name='message' )
    runner = mock_async_client.return_value.beta.messages.tool_runner.return_value
    runner.until_done = AsyncMock( return_value=expected )
    agent = create_agent( )

    result = await agent.run_async( 'Inspect the dataset.' )

    runner.until_done.assert_awaited_once_with( )
    assert result is expected
    assert agent.result is expected


@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
def test_stream_returns_streaming_tool_runner( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify streamed execution and provider-native runner retention.'''
    expected = Mock( name='streaming_runner' )
    mock_async_client.return_value.beta.messages.tool_runner.return_value = expected
    agent = create_agent( )

    result = agent.stream( 'Inspect the dataset.' )

    arguments = mock_async_client.return_value.beta.messages.tool_runner.call_args.kwargs
    assert arguments[ 'stream' ] is True
    assert arguments[ 'tools' ] == agent.async_tools
    assert result is expected
    assert agent.result is expected


@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
def test_constructor_rejects_invalid_tools( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify that Claude DataAgent accepts only Anthropic beta tools.'''
    with pytest.raises( ValueError ):
        DataAgent(
            name='Data Agent',
            model='claude-test',
            instructions='Analyze the supplied data.',
            tools=[ lambda: None ],
            api_key='test-key',
        )


@patch( 'minions.claude.agents.beta_async_tool' )
@patch( 'minions.claude.agents.AsyncAnthropic' )
@patch( 'minions.claude.agents.Anthropic' )
def test_execution_rejects_empty_prompt( mock_client: Mock,
        mock_async_client: Mock, mock_beta_async_tool: Mock ) -> None:
    '''Verify that synchronous and streaming execution reject an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        agent.run( '' )

    with pytest.raises( ValueError ):
        agent.stream( '' )

