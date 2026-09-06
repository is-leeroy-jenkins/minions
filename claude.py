'''Anthropic Claude Minion implementations.'''
from __future__ import annotations

from collections.abc import Sequence
import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic, beta_async_tool
from anthropic.lib.tools import (
    BetaAsyncFunctionTool,
    BetaAsyncStreamingToolRunner,
    BetaFunctionTool,
)
from anthropic.types.beta import BetaMessage

from . import throw_if, throw_if_less_than


class Minion:
    """Claude workflow agent backed by Anthropic's native tool runner."""


    def __init__( self, name: str, model: str, instructions: str,
            tools: Sequence[ BetaFunctionTool ] | None=None, max_turns: int=10,
            max_tokens: int=4096, api_key: str | None=None ) -> None:
        """Initialize a Claude Minion.

        Args:
            name (str): Human-readable agent name.
            model (str): Claude model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[BetaFunctionTool] | None): Optional Anthropic beta tools.
            max_turns (int): Maximum model iterations per execution.
            max_tokens (int): Maximum output tokens per model iteration.
            api_key (str | None): Optional Anthropic API key override.

        Returns:
            None: Configuration and native clients are stored.
        """
        throw_if( 'name', name )
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        throw_if_less_than( 'max_tokens', max_tokens, 1 )
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = list( tools or [ ] )
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv( 'ANTHROPIC_API_KEY' )
        throw_if( 'api_key', self.api_key )
        if not all( isinstance( tool, BetaFunctionTool ) for tool in self.tools ):
            raise TypeError( 'Argument "tools" must contain Anthropic beta tools!' )
        self.async_tools = [ self.create_async_tool( tool ) for tool in self.tools ]
        self.client = Anthropic( api_key=self.api_key )
        self.async_client = AsyncAnthropic( api_key=self.api_key )
        self.result: BetaMessage | BetaAsyncStreamingToolRunner[ object ] | None = None


    def create_async_tool( self, tool: BetaFunctionTool ) -> BetaAsyncFunctionTool:
        """Create an async adapter without changing the provider schema.

        Args:
            tool (BetaFunctionTool): Synchronous Anthropic beta tool.

        Returns:
            BetaAsyncFunctionTool: Schema-identical asynchronous tool.
        """
        async def invoke( **arguments: object ) -> object:
            return await asyncio.to_thread( tool.func, **arguments )

        invoke.__name__ = tool.name
        invoke.__doc__ = tool.description
        return beta_async_tool(
            invoke,
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )


    def run( self, prompt: str ) -> BetaMessage:
        """Execute the complete Claude workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            BetaMessage: Final provider-native message.
        """
        throw_if( 'prompt', prompt )
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.tools,
            messages=[ { 'role': 'user', 'content': prompt } ],
        )
        result = runner.until_done( )
        self.result = result
        return result


    async def run_async( self, prompt: str ) -> BetaMessage:
        """Execute the complete Claude workflow asynchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            BetaMessage: Final provider-native message.
        """
        throw_if( 'prompt', prompt )
        runner = self.async_client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.async_tools,
            messages=[ { 'role': 'user', 'content': prompt } ],
        )
        result = await runner.until_done( )
        self.result = result
        return result


    def stream( self, prompt: str ) -> BetaAsyncStreamingToolRunner[ object ]:
        """Start Anthropic's complete async streaming tool runner.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            BetaAsyncStreamingToolRunner[object]: Native runner that streams and executes tools.
        """
        throw_if( 'prompt', prompt )
        runner = self.async_client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.async_tools,
            messages=[ { 'role': 'user', 'content': prompt } ],
            stream=True,
        )
        self.result = runner
        return runner


class DataMinion( Minion ):
    """Claude Minion specialized for data workflows."""


    def __init__( self, model: str, instructions: str,
            tools: Sequence[ BetaFunctionTool ] | None=None, max_turns: int=10,
            max_tokens: int=4096, api_key: str | None=None,
            name: str='Data Minion' ) -> None:
        """Initialize a Claude data Minion.

        Args:
            model (str): Claude model identifier.
            instructions (str): Data-workflow instructions.
            tools (Sequence[BetaFunctionTool] | None): Optional Anthropic beta tools.
            max_turns (int): Maximum model iterations per execution.
            max_tokens (int): Maximum output tokens per iteration.
            api_key (str | None): Optional Anthropic API key override.
            name (str): Human-readable agent name.

        Returns:
            None: Configuration and native clients are stored.
        """
        super( ).__init__(
            name, model, instructions, tools, max_turns, max_tokens, api_key
        )


__all__: list[ str ] = [ 'DataMinion', 'Minion' ]
