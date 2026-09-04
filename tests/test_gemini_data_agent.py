'''Tests for the Google Gemini DataAgent implementation.'''
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from minions.gemini import DataAgent


def sample_tool( value: str ) -> dict[ str, str ]:
    '''Return a provider-compatible test result.'''
    return { 'value': value }


def create_agent( ) -> DataAgent:
    '''Create a DataAgent configured with a Gemini-compatible test tool.'''
    return DataAgent(
        name='Data Agent',
        model='gemini-test',
        instructions='Analyze the supplied data.',
        tools=[ sample_tool ],
        max_turns=7,
    )


async def emit( *events: Any ) -> AsyncIterator[ Any ]:
    '''Yield provider events for asynchronous execution tests.'''
    for event in events:
        yield event


@patch( 'minions.gemini.agents.InMemorySessionService' )
@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
def test_constructor_creates_google_agent( mock_agent: Mock, mock_runner: Mock,
        mock_session_service: Mock ) -> None:
    '''Verify that stored members are used to create the Google ADK runtime.'''
    agent = create_agent( )

    mock_agent.assert_called_once_with(
        name='data_agent',
        model=agent.model,
        instruction=agent.instructions,
        tools=agent.tools,
    )
    mock_runner.assert_called_once_with(
        agent=mock_agent.return_value,
        app_name='minions_data_agent',
        session_service=mock_session_service.return_value,
        auto_create_session=True,
    )
    assert agent.provider is mock_runner.return_value
    assert agent.run_config.max_llm_calls == agent.max_turns


@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
def test_run_returns_final_provider_event( mock_agent: Mock, mock_runner: Mock ) -> None:
    '''Verify synchronous execution and final ADK event selection.'''
    intermediate = Mock( )
    intermediate.is_final_response.return_value = False
    final = Mock( )
    final.is_final_response.return_value = True
    mock_runner.return_value.run.return_value = iter( [ intermediate, final ] )
    agent = create_agent( )

    result = agent.run( 'Inspect the dataset.' )

    arguments = mock_runner.return_value.run.call_args.kwargs
    assert arguments[ 'user_id' ] == agent.user_id
    assert len( arguments[ 'session_id' ] ) == 32
    assert arguments[ 'new_message' ].role == 'user'
    assert arguments[ 'run_config' ] is agent.run_config
    assert result is final
    assert agent.events == [ intermediate, final ]


@pytest.mark.asyncio
@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
async def test_run_async_returns_final_provider_event( mock_agent: Mock,
        mock_runner: Mock ) -> None:
    '''Verify asynchronous execution and final ADK event selection.'''
    intermediate = Mock( )
    intermediate.is_final_response.return_value = False
    final = Mock( )
    final.is_final_response.return_value = True
    mock_runner.return_value.run_async.return_value = emit( intermediate, final )
    agent = create_agent( )

    result = await agent.run_async( 'Inspect the dataset.' )

    assert result is final
    assert agent.events == [ intermediate, final ]


@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
def test_stream_returns_provider_stream( mock_agent: Mock, mock_runner: Mock ) -> None:
    '''Verify streamed execution and provider-native stream retention.'''
    expected = emit( Mock( ) )
    mock_runner.return_value.run_async.return_value = expected
    agent = create_agent( )

    result = agent.stream( 'Inspect the dataset.' )

    assert result is expected
    assert agent.result is expected


@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
def test_constructor_rejects_noncallable_tools( mock_agent: Mock,
        mock_runner: Mock ) -> None:
    '''Verify that Gemini DataAgent accepts only executable callable tools.'''
    with pytest.raises( ValueError ):
        DataAgent(
            name='Data Agent',
            model='gemini-test',
            instructions='Analyze the supplied data.',
            tools=[ { 'type': 'function' } ],
        )


@patch( 'minions.gemini.agents.Runner' )
@patch( 'minions.gemini.agents.Agent' )
def test_execution_rejects_empty_prompt( mock_agent: Mock, mock_runner: Mock ) -> None:
    '''Verify that synchronous and streaming execution reject an empty prompt.'''
    agent = create_agent( )

    with pytest.raises( ValueError ):
        agent.run( '' )

    with pytest.raises( ValueError ):
        agent.stream( '' )

