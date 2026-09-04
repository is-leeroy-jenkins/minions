'''Tests for the OpenAI DataAgent implementation.'''
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minions.gpt import DataAgent


def create_agent( ) -> DataAgent:
    '''Create a DataAgent configured with a provider-compatible test tool.'''
    return DataAgent(
        name='Data Agent',
        model='gpt-test',
        instructions='Analyze the supplied data.',
        tools=[ Mock( name='tool' ) ],
        max_turns=7,
    )


@patch( 'minions.gpt.agents.Agent' )
def test_constructor_creates_openai_agent( mock_agent: Mock ) -> None:
    '''Verify that stored members are used to create the OpenAI agent.'''
    tool = Mock( name='tool' )

    agent = DataAgent(
        name='Data Agent',
        model='gpt-test',
        instructions='Analyze the supplied data.',
        tools=[ tool ],
        max_turns=7,
    )

    mock_agent.assert_called_once_with(
        name=agent.name,
        model=agent.model,
        instructions=agent.instructions,
        tools=agent.tools,
    )
    assert agent.provider is mock_agent.return_value


@patch( 'minions.gpt.agents.Runner.run_sync' )
@patch( 'minions.gpt.agents.Agent' )
def test_run_returns_provider_result( mock_agent: Mock, mock_run_sync: Mock ) -> None:
    '''Verify synchronous execution and provider-native result retention.'''
    expected = Mock( name='run_result' )
    mock_run_sync.return_value = expected
    agent = create_agent( )

    result = agent.run( 'Inspect the dataset.' )

    mock_run_sync.assert_called_once_with(
        starting_agent=agent.provider,
        input=agent.prompt,
        max_turns=agent.max_turns,
    )
    assert result is expected
    assert agent.result is expected


@pytest.mark.asyncio
@patch( 'minions.gpt.agents.Runner.run', new_callable=AsyncMock )
@patch( 'minions.gpt.agents.Agent' )
async def test_run_async_returns_provider_result( mock_agent: Mock,
        mock_run: AsyncMock ) -> None:
    '''Verify asynchronous execution and provider-native result retention.'''
    expected = Mock( name='run_result' )
    mock_run.return_value = expected
    agent = create_agent( )

    result = await agent.run_async( 'Inspect the dataset.' )

    mock_run.assert_awaited_once_with(
        starting_agent=agent.provider,
        input=agent.prompt,
        max_turns=agent.max_turns,
    )
    assert result is expected
    assert agent.result is expected


@patch( 'minions.gpt.agents.Runner.run_streamed' )
@patch( 'minions.gpt.agents.Agent' )
def test_stream_returns_provider_stream( mock_agent: Mock, mock_run_streamed: Mock ) -> None:
    '''Verify streamed execution and provider-native stream retention.'''
    expected = Mock( name='streaming_result' )
    mock_run_streamed.return_value = expected
    agent = create_agent( )

    result = agent.stream( 'Inspect the dataset.' )

    mock_run_streamed.assert_called_once_with(
        starting_agent=agent.provider,
        input=agent.prompt,
        max_turns=agent.max_turns,
    )
    assert result is expected
    assert agent.result is expected


@patch( 'minions.gpt.agents.Agent' )
def test_execution_rejects_empty_prompt( mock_agent: Mock ) -> None:
    '''Verify that synchronous and streaming execution reject an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        agent.run( '' )

    with pytest.raises( ValueError ):
        agent.stream( '' )


@pytest.mark.asyncio
@patch( 'minions.gpt.agents.Agent' )
async def test_async_execution_rejects_empty_prompt( mock_agent: Mock ) -> None:
    '''Verify that asynchronous execution rejects an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        await agent.run_async( '' )
