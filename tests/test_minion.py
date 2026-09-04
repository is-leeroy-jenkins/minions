'''Tests for the shared Minion contract.'''
from __future__ import annotations

from typing import Any

import pytest

from minions import Minion


class ExampleAgent( Minion ):
    '''Concrete Minion used to test the shared contract.'''


    def create( self ) -> object:
        self.provider = object( )
        return self.provider


    def run( self, prompt: str ) -> Any:
        self.prompt = prompt
        self.result = prompt
        return self.result


    async def run_async( self, prompt: str ) -> Any:
        self.prompt = prompt
        self.result = prompt
        return self.result


    def stream( self, prompt: str ) -> Any:
        self.prompt = prompt
        self.result = prompt
        return self.result


@pytest.mark.parametrize(
    ( 'name', 'model', 'instructions', 'tools', 'max_turns' ),
    [
        ( '', 'gpt-test', 'Analyze the supplied data.', [ object( ) ], 10 ),
        ( 'Data Agent', '', 'Analyze the supplied data.', [ object( ) ], 10 ),
        ( 'Data Agent', 'gpt-test', '', [ object( ) ], 10 ),
        ( 'Data Agent', 'gpt-test', 'Analyze the supplied data.', [ ], 10 ),
        ( 'Data Agent', 'gpt-test', 'Analyze the supplied data.', [ object( ) ], 0 ),
    ],
)
def test_minion_rejects_empty_required_arguments( name: str, model: str, instructions: str,
        tools: list[ Any ], max_turns: int ) -> None:
    '''Verify that every required constructor argument is guarded.'''
    with pytest.raises( ValueError ):
        ExampleAgent( name, model, instructions, tools, max_turns )


def test_reset_clears_only_transient_state( ) -> None:
    '''Verify that reset preserves permanent configuration and the provider runtime.'''
    tool = object( )
    agent = ExampleAgent(
        name='Data Agent',
        model='gpt-test',
        instructions='Analyze the supplied data.',
        tools=[ tool ],
    )
    provider = agent.create( )
    agent.run( 'Inspect the dataset.' )

    agent.reset( )

    assert agent.name == 'Data Agent'
    assert agent.model == 'gpt-test'
    assert agent.instructions == 'Analyze the supplied data.'
    assert agent.tools == [ tool ]
    assert agent.provider is provider
    assert agent.prompt == ''
    assert agent.result is None
