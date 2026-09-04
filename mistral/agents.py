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
    Implements data-oriented workflow agents using the Mistral AI SDK.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import Callable
from inspect import isawaitable
from typing import Any
import json
import os

from mistralai.client import Mistral
from mistralai.extra.run.tools import create_tool_call

from minions import Minion, throw_if


class DataAgent( Minion ):
    """Mistral AI data workflow agent.

    Purpose:
        Creates and executes a data-oriented workflow agent with the Mistral AI SDK. Tools
        supplied to this class must be callable Mistral tools, including callable operations from
        ``fonky.mistral.tools``.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): Mistral model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Callable[..., Any]]): Mistral-compatible callable tools.
        max_turns (int): Maximum number of model turns allowed for one execution.
        api_key (str | None): Optional Mistral API key overriding ``MISTRAL_API_KEY``.
    """


    def __init__( self, name: str, model: str, instructions: str,
            tools: list[ Callable[ ..., Any ] ], max_turns: int=10,
            api_key: str | None=None ) -> None:
        """Initialize the Mistral AI data workflow agent.

        Purpose:
            Validates and stores workflow configuration, creates the Mistral client, converts
            callable tools into provider-native schemas, and creates the provider agent.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): Mistral model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Callable[..., Any]]): Mistral-compatible callable tools.
            max_turns (int): Maximum number of model turns allowed for one execution.
            api_key (str | None): Optional Mistral API key overriding ``MISTRAL_API_KEY``.

        Returns:
            None: Initialization creates and stores the Mistral client and agent.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )
        self.api_key = api_key or os.getenv( 'MISTRAL_API_KEY' )
        throw_if( 'api_key', self.api_key )

        if not all( callable( tool ) for tool in self.tools ):
            raise ValueError( 'Argument "tools" must contain only callable tools!' )

        self.client = Mistral( api_key=self.api_key )
        self.schemas = [ create_tool_call( tool ) for tool in self.tools ]
        self.functions = { tool.__name__: tool for tool in self.tools }
        self.provider = self.create( )


    def create( self ) -> Any:
        """Create the Mistral AI agent.

        Purpose:
            Creates a persistent Mistral agent from the validated workflow configuration and
            provider-native function schemas.

        Returns:
            Any: Mistral AI agent returned by the provider SDK.
        """
        self.provider = self.client.beta.agents.create(
            name=self.name,
            model=self.model,
            instructions=self.instructions,
            tools=self.schemas,
        )
        return self.provider


    def run( self, prompt: str ) -> Any:
        """Run the Mistral AI data workflow synchronously.

        Purpose:
            Executes model turns, runs requested local tools, and submits their results until the
            model returns a final response or the configured turn limit is reached.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Mistral AI completion response.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        messages: list[ Any ] = [ { 'role': 'user', 'content': self.prompt } ]

        for turn in range( self.max_turns ):
            self.result = self.client.agents.complete(
                agent_id=self.provider.id,
                messages=messages,
            )
            tool_calls = self.result.choices[ 0 ].message.tool_calls or [ ]
            if not tool_calls:
                return self.result

            messages.append( self.result.choices[ 0 ].message )
            messages.extend( self.execute_tools( tool_calls ) )

        raise RuntimeError(
            f'Mistral workflow exceeded the maximum of {self.max_turns} model turns!'
        )


    async def run_async( self, prompt: str ) -> Any:
        """Run the Mistral AI data workflow asynchronously.

        Purpose:
            Executes asynchronous model turns, runs requested local tools, and submits their
            results until the model returns a final response or the turn limit is reached.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Mistral AI completion response.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        messages: list[ Any ] = [ { 'role': 'user', 'content': self.prompt } ]

        for turn in range( self.max_turns ):
            self.result = await self.client.agents.complete_async(
                agent_id=self.provider.id,
                messages=messages,
            )
            tool_calls = self.result.choices[ 0 ].message.tool_calls or [ ]
            if not tool_calls:
                return self.result

            messages.append( self.result.choices[ 0 ].message )
            messages.extend( await self.execute_tools_async( tool_calls ) )

        raise RuntimeError(
            f'Mistral workflow exceeded the maximum of {self.max_turns} model turns!'
        )


    def stream( self, prompt: str ) -> Any:
        """Start a streamed Mistral AI workflow turn.

        Purpose:
            Starts a provider-native Mistral agent stream. Tool-call events remain available in
            the returned stream for caller-controlled streamed execution.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Mistral AI server-sent event stream.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = self.client.agents.stream(
            agent_id=self.provider.id,
            messages=[ { 'role': 'user', 'content': self.prompt } ],
        )
        return self.result


    def execute_tools( self, tool_calls: list[ Any ] ) -> list[ dict[ str, Any ] ]:
        """Execute synchronous Mistral function calls.

        Purpose:
            Resolves each provider function call to its registered callable and creates the tool
            messages required for the next model turn.

        Args:
            tool_calls (list[Any]): Mistral function calls returned by a model turn.

        Returns:
            list[dict[str, Any]]: Tool-result messages for the next model turn.
        """
        throw_if( 'tool_calls', tool_calls )
        messages: list[ dict[ str, Any ] ] = [ ]

        for tool_call in tool_calls:
            name = tool_call.function.name
            function = self.functions.get( name )
            if function is None:
                raise ValueError( f'Mistral requested an unregistered tool: "{name}"!' )

            arguments = tool_call.function.arguments
            values = json.loads( arguments ) if isinstance( arguments, str ) else arguments
            output = function( **values )
            if isawaitable( output ):
                raise TypeError( f'Tool "{name}" requires asynchronous execution!' )

            messages.append( {
                'role': 'tool',
                'name': name,
                'content': self.serialize_result( output ),
                'tool_call_id': tool_call.id,
            } )

        return messages


    async def execute_tools_async( self,
            tool_calls: list[ Any ] ) -> list[ dict[ str, Any ] ]:
        """Execute asynchronous Mistral function calls.

        Purpose:
            Resolves provider function calls to registered callables, awaits asynchronous results
            when required, and creates tool messages for the next model turn.

        Args:
            tool_calls (list[Any]): Mistral function calls returned by a model turn.

        Returns:
            list[dict[str, Any]]: Tool-result messages for the next model turn.
        """
        throw_if( 'tool_calls', tool_calls )
        messages: list[ dict[ str, Any ] ] = [ ]

        for tool_call in tool_calls:
            name = tool_call.function.name
            function = self.functions.get( name )
            if function is None:
                raise ValueError( f'Mistral requested an unregistered tool: "{name}"!' )

            arguments = tool_call.function.arguments
            values = json.loads( arguments ) if isinstance( arguments, str ) else arguments
            output = function( **values )
            if isawaitable( output ):
                output = await output

            messages.append( {
                'role': 'tool',
                'name': name,
                'content': self.serialize_result( output ),
                'tool_call_id': tool_call.id,
            } )

        return messages


    def serialize_result( self, result: Any ) -> str:
        """Serialize a local tool result for Mistral.

        Purpose:
            Converts provider tool output into the string content required by a Mistral tool
            message while retaining structured JSON whenever possible.

        Args:
            result (Any): Value returned by a registered local tool.

        Returns:
            str: Serialized tool-result content.
        """
        if isinstance( result, str ):
            return result
        return json.dumps( result, default=str )
