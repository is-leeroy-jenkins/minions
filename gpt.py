'''OpenAI Agents SDK Minion implementations.'''
from __future__ import annotations

from collections.abc import Sequence

from agents import Agent, RunResult, RunResultStreaming, Runner, Tool

from . import throw_if, throw_if_less_than


class Minion( Agent ):
    """Provider-native OpenAI workflow agent."""

    max_turns: int
    result: RunResult | RunResultStreaming | None


    def __init__( self, name: str, model: str, instructions: str,
            tools: Sequence[ Tool ] | None=None, max_turns: int=10 ) -> None:
        """Initialize an OpenAI Minion.

        Args:
            name (str): Human-readable agent name.
            model (str): OpenAI model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[Tool] | None): Optional OpenAI Agents SDK tools.
            max_turns (int): Maximum model turns per execution.

        Returns:
            None: Configuration is stored on the provider-native agent.
        """
        throw_if( 'name', name )
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        super( ).__init__(
            name=name,
            model=model,
            instructions=instructions,
            tools=list( tools or [ ] ),
        )
        self.max_turns = max_turns
        self.result = None


    def run( self, prompt: str ) -> RunResult:
        """Execute the complete OpenAI workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            RunResult: Provider-native completed run.
        """
        throw_if( 'prompt', prompt )
        result = Runner.run_sync( self, prompt, max_turns=self.max_turns )
        self.result = result
        return result


    async def run_async( self, prompt: str ) -> RunResult:
        """Execute the complete OpenAI workflow asynchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            RunResult: Provider-native completed run.
        """
        throw_if( 'prompt', prompt )
        result = await Runner.run( self, prompt, max_turns=self.max_turns )
        self.result = result
        return result


    def stream( self, prompt: str ) -> RunResultStreaming:
        """Start a provider-managed streamed workflow.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            RunResultStreaming: Native stream with automatic tool handling.
        """
        throw_if( 'prompt', prompt )
        result = Runner.run_streamed( self, prompt, max_turns=self.max_turns )
        self.result = result
        return result


class DataMinion( Minion ):
    """OpenAI Minion specialized for data workflows."""


    def __init__( self, model: str, instructions: str, tools: Sequence[ Tool ] | None=None,
            max_turns: int=10, name: str='Data Minion' ) -> None:
        """Initialize an OpenAI data Minion.

        Args:
            model (str): OpenAI model identifier.
            instructions (str): Data-workflow instructions.
            tools (Sequence[Tool] | None): Optional OpenAI Agents SDK tools.
            max_turns (int): Maximum model turns per execution.
            name (str): Human-readable agent name.

        Returns:
            None: Configuration is stored on the provider-native agent.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )


__all__: list[ str ] = [ 'DataMinion', 'Minion' ]
