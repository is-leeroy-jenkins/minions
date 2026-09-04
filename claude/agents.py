'''
  ******************************************************************************************
      Assembly:                Minions
      Filename:                agents.py
      Author:                  Terry D. Eppler
      Created:                 09-04-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-04-2026
  ******************************************************************************************
  <summary>
    Implements data-oriented workflow agents using the Anthropic Python SDK.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any
import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic, beta_async_tool

from minions import Minion, throw_if


class DataAgent( Minion ):
    """Anthropic Claude data workflow agent.

    Purpose:
        Creates and executes data-oriented Claude workflows with Anthropic's automatic tool
        runner. Tools from ``fonky.claude.tools`` retain their provider schemas for synchronous,
        asynchronous, and streamed execution.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): Claude model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Any]): Anthropic ``@beta_tool`` objects.
        max_turns (int): Maximum number of model turns allowed for one execution.
        max_tokens (int): Maximum number of output tokens allowed for each model turn.
        api_key (str | None): Optional Anthropic API key overriding ``ANTHROPIC_API_KEY``.
    """


    def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ],
            max_turns: int=10, max_tokens: int=4096, api_key: str | None=None ) -> None:
        """Initialize the Anthropic Claude data workflow agent.

        Purpose:
            Validates Anthropic function tools and execution limits, creates synchronous and
            asynchronous clients, and derives schema-identical asynchronous tool adapters.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): Claude model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Any]): Anthropic ``@beta_tool`` objects.
            max_turns (int): Maximum number of model turns allowed for one execution.
            max_tokens (int): Maximum number of output tokens allowed for each model turn.
            api_key (str | None): Optional Anthropic API key overriding ``ANTHROPIC_API_KEY``.

        Returns:
            None: Initialization creates and stores the Anthropic clients and tool adapters.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )
        self.max_tokens = max_tokens
        throw_if( 'max_tokens', self.max_tokens )
        if self.max_tokens < 1:
            raise ValueError( 'Argument "max_tokens" must be greater than zero!' )

        if not all( self.is_tool( tool ) for tool in self.tools ):
            raise ValueError( 'Argument "tools" must contain only Anthropic beta tools!' )

        self.api_key = api_key or os.getenv( 'ANTHROPIC_API_KEY' )
        throw_if( 'api_key', self.api_key )
        self.async_tools = [ self.create_async_tool( tool ) for tool in self.tools ]
        self.client: Anthropic | None = None
        self.async_client: AsyncAnthropic | None = None
        self.provider = self.create( )


    def is_tool( self, tool: Any ) -> bool:
        """Determine whether a value is an Anthropic function tool.

        Purpose:
            Confirms that a supplied tool exposes the schema, name, callable, and execution members
            required by Anthropic's beta tool runner.

        Args:
            tool (Any): Candidate Anthropic tool object.

        Returns:
            bool: ``True`` when the required Anthropic tool contract is present.
        """
        return (
            callable( getattr( tool, 'to_dict', None ) )
            and callable( getattr( tool, 'call', None ) )
            and callable( getattr( tool, 'func', None ) )
            and bool( getattr( tool, 'name', '' ) )
            and bool( getattr( tool, 'input_schema', None ) )
        )


    def create_async_tool( self, tool: Any ) -> Any:
        """Create an asynchronous Anthropic tool adapter.

        Purpose:
            Preserves the Fonky ``@beta_tool`` name, description, and input schema while executing
            its synchronous callable on a worker thread for Anthropic's native async tool runner.

        Args:
            tool (Any): Validated Anthropic synchronous function tool.

        Returns:
            Any: Anthropic ``BetaAsyncFunctionTool`` instance.
        """
        if not self.is_tool( tool ):
            raise ValueError( 'Argument "tool" must be an Anthropic beta tool!' )

        async def invoke( **arguments: Any ) -> Any:
            return await asyncio.to_thread( tool.func, **arguments )

        invoke.__name__ = tool.name
        invoke.__doc__ = tool.description
        return beta_async_tool(
            invoke,
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )


    def create( self ) -> Anthropic:
        """Create synchronous and asynchronous Anthropic clients.

        Purpose:
            Creates provider-native Anthropic clients using the validated API credential.

        Returns:
            Anthropic: Synchronous Anthropic client retained as the primary provider runtime.
        """
        self.client = Anthropic( api_key=self.api_key )
        self.async_client = AsyncAnthropic( api_key=self.api_key )
        self.provider = self.client
        return self.provider


    def run( self, prompt: str ) -> Any:
        """Run the Anthropic Claude data workflow synchronously.

        Purpose:
            Executes the complete bounded Anthropic tool-runner loop and returns the final
            provider-native Claude message.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Anthropic ``BetaMessage``.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.tools,
            messages=[ { 'role': 'user', 'content': self.prompt } ],
        )
        self.result = runner.until_done( )
        return self.result


    async def run_async( self, prompt: str ) -> Any:
        """Run the Anthropic Claude data workflow asynchronously.

        Purpose:
            Executes Anthropic's native asynchronous tool-runner loop with schema-identical async
            Fonky tools and returns the final provider-native Claude message.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Anthropic ``BetaMessage``.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        runner = self.async_client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.async_tools,
            messages=[ { 'role': 'user', 'content': self.prompt } ],
        )
        self.result = await runner.until_done( )
        return self.result


    def stream( self, prompt: str ) -> Any:
        """Start a streamed Anthropic Claude data workflow.

        Purpose:
            Returns Anthropic's native asynchronous streaming tool runner. The runner streams each
            model turn, executes requested Fonky tools, and continues until Claude finishes.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Anthropic ``BetaAsyncStreamingToolRunner``.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = self.async_client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_turns,
            system=self.instructions,
            tools=self.async_tools,
            messages=[ { 'role': 'user', 'content': self.prompt } ],
            stream=True,
        )
        return self.result
