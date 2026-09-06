'''Architecture and validation tests for every provider pathway.'''
from __future__ import annotations

from importlib import import_module
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


@pytest.mark.parametrize( 'provider', [ 'gpt', 'gemini', 'grok', 'claude', 'mistral' ] )
def test_every_provider_exports_the_complete_concrete_minion_family(
        provider: str ) -> None:
    '''Verify all concrete workflow implementations exist and inherit provider Minion.'''
    module = import_module( f'minions.{provider}' )
    expected = {
        'CodingMinion': 'Coding Minion',
        'DataMinion': 'Data Minion',
        'PlanningMinion': 'Planning Minion',
        'ResearchMinion': 'Research Minion',
        'ReviewMinion': 'Review Minion',
        'WritingMinion': 'Writing Minion',
    }

    for class_name, default_name in expected.items( ):
        implementation = getattr( module, class_name )
        assert issubclass( implementation, module.Minion )
        assert implementation.minion_name == default_name


@pytest.mark.parametrize( 'provider', [ 'gpt', 'gemini', 'grok', 'claude', 'mistral' ] )
def test_provider_modules_do_not_expose_factory_functions( provider: str ) -> None:
    '''Verify concrete classes are the only construction mechanism.'''
    module = import_module( f'minions.{provider}' )
    assert not hasattr( module, 'create_minion' )


@pytest.mark.parametrize( 'provider', [ 'gpt', 'gemini' ] )
def test_native_concrete_minions_are_constructible( provider: str ) -> None:
    '''Verify every native Agent specialization constructs with its workflow name.'''
    module = import_module( f'minions.{provider}' )
    names = [
        'CodingMinion',
        'DataMinion',
        'PlanningMinion',
        'ResearchMinion',
        'ReviewMinion',
        'WritingMinion',
    ]

    for class_name in names:
        implementation = getattr( module, class_name )
        minion = implementation( model=f'{provider}-test', instructions='Do the work.' )
        actual_name = getattr( minion, 'display_name', minion.name )
        assert actual_name == implementation.minion_name


@pytest.mark.parametrize( 'provider', [ 'grok', 'claude', 'mistral' ] )
def test_wrapped_concrete_minions_are_constructible( provider: str ) -> None:
    '''Verify every wrapped provider specialization constructs with its workflow name.'''
    module = import_module( f'minions.{provider}' )
    names = [
        'CodingMinion',
        'DataMinion',
        'PlanningMinion',
        'ResearchMinion',
        'ReviewMinion',
        'WritingMinion',
    ]
    patches = {
        'grok': [ patch( 'minions.grok.Client' ), patch( 'minions.grok.AsyncClient' ) ],
        'claude': [
            patch( 'minions.claude.Anthropic' ),
            patch( 'minions.claude.AsyncAnthropic' ),
        ],
        'mistral': [ patch( 'minions.mistral.Mistral' ) ],
    }

    for provider_patch in patches[ provider ]:
        provider_patch.start( )
    try:
        for class_name in names:
            implementation = getattr( module, class_name )
            minion = implementation(
                model=f'{provider}-test', instructions='Do the work.', api_key='test-key'
            )
            assert minion.name == implementation.minion_name
    finally:
        for provider_patch in reversed( patches[ provider ] ):
            provider_patch.stop( )
