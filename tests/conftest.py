'''Shared test configuration.'''
from __future__ import annotations

from importlib.util import find_spec
from types import ModuleType
from typing import Any
import sys


if find_spec( 'agents' ) is None:
    agents = ModuleType( 'agents' )

    class Agent:
        '''Minimal OpenAI Agent substitute used when the SDK is unavailable locally.'''


        def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ] ) -> None:
            self.name = name
            self.model = model
            self.instructions = instructions
            self.tools = tools


    class Runner:
        '''Minimal OpenAI Runner substitute replaced by mocks in each execution test.'''


        @classmethod
        async def run( cls, starting_agent: Agent, input: str, max_turns: int ) -> Any:
            raise NotImplementedError


        @classmethod
        def run_sync( cls, starting_agent: Agent, input: str, max_turns: int ) -> Any:
            raise NotImplementedError


        @classmethod
        def run_streamed( cls, starting_agent: Agent, input: str, max_turns: int ) -> Any:
            raise NotImplementedError


    agents.Agent = Agent
    agents.Runner = Runner
    sys.modules[ 'agents' ] = agents
