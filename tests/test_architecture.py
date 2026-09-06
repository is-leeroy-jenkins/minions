'''Architecture and validation tests for every provider pathway.'''
from __future__ import annotations

from unittest.mock import patch

import pytest
from agents import Agent as OpenAIAgent
from google.adk import Agent as GeminiAgent

from minions.claude import DataMinion as ClaudeDataMinion
from minions.claude import Minion as ClaudeMinion
from minions.gemini import DataMinion as GeminiDataMinion
from minions.gemini import Minion as GeminiMinion
from minions.gpt import DataMinion as GptDataMinion
from minions.gpt import Minion as GptMinion
from minions.grok import DataMinion as GrokDataMinion
from minions.grok import Minion as GrokMinion
from minions.mistral import DataMinion as MistralDataMinion
from minions.mistral import Minion as MistralMinion


def test_provider_minions_use_native_inheritance_when_available( ) -> None:
    '''Verify direct inheritance for SDKs that expose a suitable Agent class.'''
    assert issubclass( GptMinion, OpenAIAgent )
    assert issubclass( GeminiMinion, GeminiAgent )


@pytest.mark.parametrize(
    ( 'implementation', 'base' ),
    [
        ( GptDataMinion, GptMinion ),
        ( GeminiDataMinion, GeminiMinion ),
        ( GrokDataMinion, GrokMinion ),
        ( ClaudeDataMinion, ClaudeMinion ),
        ( MistralDataMinion, MistralMinion ),
    ],
)
def test_every_data_minion_inherits_its_provider_minion(
        implementation: type[ object ], base: type[ object ] ) -> None:
    '''Verify complete provider coverage and the requested naming model.'''
    assert issubclass( implementation, base )


def test_openai_and_gemini_tools_are_optional( ) -> None:
    '''Verify tool-free agents are valid for providers with local native agents.'''
    gpt = GptDataMinion( model='gpt-test', instructions='Help the user.' )
    gemini = GeminiDataMinion( model='gemini-test', instructions='Help the user.' )

    assert gpt.tools == [ ]
    assert gemini.tools == [ ]


@patch( 'minions.grok.AsyncClient' )
@patch( 'minions.grok.Client' )
def test_grok_tools_are_optional( client: object, async_client: object ) -> None:
    '''Verify Grok accepts a workflow without tools or functions.'''
    minion = GrokDataMinion(
        model='grok-test', instructions='Help the user.', api_key='test-key'
    )
    assert minion.tools == [ ]
    assert minion.functions == { }


@patch( 'minions.claude.AsyncAnthropic' )
@patch( 'minions.claude.Anthropic' )
def test_claude_tools_are_optional( client: object, async_client: object ) -> None:
    '''Verify Claude accepts a workflow without tools.'''
    minion = ClaudeDataMinion(
        model='claude-test', instructions='Help the user.', api_key='test-key'
    )
    assert minion.tools == [ ]
    assert minion.async_tools == [ ]


@patch( 'minions.mistral.Mistral' )
def test_mistral_tools_are_optional( client: object ) -> None:
    '''Verify Mistral accepts a workflow without tools or functions.'''
    minion = MistralDataMinion(
        model='mistral-test', instructions='Help the user.', api_key='test-key'
    )
    assert minion.tools == [ ]
    assert minion.functions == { }


def test_root_does_not_export_a_cross_provider_minion( ) -> None:
    '''Verify the artificial universal ABC has been removed.'''
    import minions

    assert not hasattr( minions, 'Minion' )
