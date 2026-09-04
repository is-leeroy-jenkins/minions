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
    Implements data-oriented workflow agents using the xAI Python SDK.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from inspect import isawaitable, iscoroutinefunction
from typing import Any
import asyncio
import json
import os

from xai_sdk import AsyncClient, Client
from xai_sdk.chat import system, tool_result, user

from minions import Minion, throw_if


class DataAgent( Minion ):
    """xAI Grok data workflow agent.

    Purpose:
        Creates and executes data-oriented Grok chat workflows with xAI client-side tool calling.
        Fonky Grok schemas are paired explicitly with their matching executable callables so every
        requested tool can be executed and correlated with its provider call identifier.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): Grok model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Any]): xAI tool schemas, including declarations from ``fonky.grok.tools``.
        functions (list[Callable[..., Any]]): Executable Fonky callables matching the tool schemas.
        max_turns (int): Maximum number of model turns allowed for one execution.
        api_key (str | None): Optional xAI API key overriding ``XAI_API_KEY``.
    """


    def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ],
            functions: list[ Callable[ ..., Any ] ], max_turns: int=10,
            api_key: str | None=None ) -> None:
        """Initialize the xAI Grok data workflow agent.

        Purpose:
            Validates provider schemas and executable functions, creates synchronous and
            asynchronous xAI clients, and verifies an exact schema-to-callable mapping.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): Grok model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Any]): xAI tool schemas, including declarations from ``fonky.grok.tools``.
            functions (list[Callable[..., Any]]): Executable callables matching the tool schemas.
            max_turns (int): Maximum number of model turns allowed for one execution.
            api_key (str | None): Optional xAI API key overriding ``XAI_API_KEY``.

        Returns:
            None: Initialization creates and stores the xAI clients.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )
        self.function_tools = functions
        throw_if( 'functions', self.function_tools )

        if not all( callable( function ) for function in self.function_tools ):
            raise ValueError( 'Argument "functions" must contain only callable Grok tools!' )

        self.functions = { function.__name__: function for function in self.function_tools }
        if len( self.functions ) != len( self.function_tools ):
            raise ValueError( 'Argument "functions" must contain unique function names!' )

        self.tool_names = [ self.get_tool_name( tool ) for tool in self.tools ]
        if len( set( self.tool_names ) ) != len( self.tool_names ):
            raise ValueError( 'Argument "tools" must contain unique xAI tool names!' )

        missing = sorted( set( self.tool_names ).difference( self.functions ) )
        extra = sorted( set( self.functions ).difference( self.tool_names ) )
        if missing or extra:
            raise ValueError(
                f'Grok tool schemas and functions must match exactly; missing={missing}, extra={extra}!'
            )

        self.api_key = api_key or os.getenv( 'XAI_API_KEY' )
        throw_if( 'api_key', self.api_key )
        self.client: Client | None = None
        self.async_client: AsyncClient | None = None
        self.provider = self.create( )


    def get_tool_name( self, tool: Any ) -> str:
        """Read an xAI client-side tool name.

        Purpose:
            Extracts and validates the function name stored by the provider-native xAI tool
            schema.

        Args:
            tool (Any): xAI tool schema.

        Returns:
            str: Provider function name.
        """
        throw_if( 'tool', tool )
        function = getattr( tool, 'function', None )
        name = getattr( function, 'name', '' )
        throw_if( 'tool.function.name', name )
        return str( name )


    def create( self ) -> Client:
        """Create synchronous and asynchronous xAI clients.

        Purpose:
            Creates provider-native xAI clients using the validated API credential.

        Returns:
            Client: Synchronous xAI client retained as the primary provider runtime.
        """
        self.client = Client( api_key=self.api_key )
        self.async_client = AsyncClient( api_key=self.api_key )
        self.provider = self.client
        return self.provider


    def create_chat( self, client: Any ) -> Any:
        """Create a provider-native Grok chat.

        Purpose:
            Creates an isolated chat with the stored model, instructions, and xAI tool schemas.

        Args:
            client (Any): Synchronous or asynchronous xAI client.

        Returns:
            Any: Provider-native xAI chat instance.
        """
        throw_if( 'client', client )
        return client.chat.create(
            model=self.model,
            messages=[ system( self.instructions ) ],
            tools=self.tools,
        )


    def run( self, prompt: str ) -> Any:
        """Run the xAI Grok data workflow synchronously.

        Purpose:
            Executes Grok model turns, runs every requested client-side Fonky tool, and submits
            correlated tool results until a final response is returned.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final provider-native xAI response.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        chat = self.create_chat( self.client )
        chat.append( user( self.prompt ) )

        for turn in range( self.max_turns ):
            self.result = chat.sample( )
            chat.append( self.result )
            tool_calls = self.result.tool_calls or [ ]
            if not tool_calls:
                return self.result
            self.execute_tools( chat, tool_calls )

        raise RuntimeError(
            f'Grok workflow exceeded the maximum of {self.max_turns} model turns!'
        )


    async def run_async( self, prompt: str ) -> Any:
        """Run the xAI Grok data workflow asynchronously.

        Purpose:
            Executes asynchronous Grok model turns and client-side Fonky tools until a final
            provider-native response is returned or the configured turn limit is reached.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final provider-native xAI response.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        chat = self.create_chat( self.async_client )
        chat.append( user( self.prompt ) )

        for turn in range( self.max_turns ):
            self.result = await chat.sample( )
            chat.append( self.result )
            tool_calls = self.result.tool_calls or [ ]
            if not tool_calls:
                return self.result
            await self.execute_tools_async( chat, tool_calls )

        raise RuntimeError(
            f'Grok workflow exceeded the maximum of {self.max_turns} model turns!'
        )


    def stream( self, prompt: str ) -> AsyncIterator[ tuple[ Any, Any ] ]:
        """Start a streamed xAI Grok data workflow.

        Purpose:
            Returns an asynchronous provider-native event stream that continues across client-side
            tool calls and stops only after Grok returns a final response.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            AsyncIterator[tuple[Any, Any]]: xAI ``(response, chunk)`` stream entries.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = self.stream_events( )
        return self.result


    async def stream_events( self ) -> AsyncIterator[ tuple[ Any, Any ] ]:
        """Execute the complete streamed Grok tool loop.

        Purpose:
            Streams each xAI response turn, executes requested client-side tools, appends correlated
            results, and resumes streaming until completion.

        Yields:
            tuple[Any, Any]: Provider-native xAI response and incremental chunk.
        """
        chat = self.create_chat( self.async_client )
        chat.append( user( self.prompt ) )

        for turn in range( self.max_turns ):
            response = None
            async for response, chunk in chat.stream( ):
                self.result = response
                yield response, chunk

            if response is None:
                raise RuntimeError( 'Grok streaming returned no provider response!' )

            chat.append( response )
            tool_calls = response.tool_calls or [ ]
            if not tool_calls:
                return
            await self.execute_tools_async( chat, tool_calls )

        raise RuntimeError(
            f'Grok workflow exceeded the maximum of {self.max_turns} model turns!'
        )


    def execute_tools( self, chat: Any, tool_calls: list[ Any ] ) -> None:
        """Execute synchronous xAI client-side tool calls.

        Purpose:
            Resolves every provider tool call, validates its JSON arguments, executes the matching
            Fonky callable, and appends a correlated tool-result message.

        Args:
            chat (Any): Provider-native xAI chat receiving tool results.
            tool_calls (list[Any]): Client-side tool calls returned by Grok.

        Returns:
            None: Tool-result messages are appended to the chat.
        """
        throw_if( 'chat', chat )
        throw_if( 'tool_calls', tool_calls )

        for tool_call in tool_calls:
            name = tool_call.function.name
            function = self.functions.get( name )
            if function is None:
                raise ValueError( f'Grok requested an unregistered tool: "{name}"!' )
            values = json.loads( tool_call.function.arguments )
            output = function( **values )
            if isawaitable( output ):
                raise TypeError( f'Tool "{name}" requires asynchronous execution!' )
            chat.append( tool_result(
                self.serialize_result( output ),
                tool_call_id=tool_call.id,
            ) )


    async def execute_tools_async( self, chat: Any, tool_calls: list[ Any ] ) -> None:
        """Execute asynchronous xAI client-side tool calls.

        Purpose:
            Executes coroutine tools directly and synchronous Fonky tools on worker threads before
            appending correlated provider-native tool-result messages.

        Args:
            chat (Any): Provider-native xAI chat receiving tool results.
            tool_calls (list[Any]): Client-side tool calls returned by Grok.

        Returns:
            None: Tool-result messages are appended to the chat.
        """
        throw_if( 'chat', chat )
        throw_if( 'tool_calls', tool_calls )

        for tool_call in tool_calls:
            name = tool_call.function.name
            function = self.functions.get( name )
            if function is None:
                raise ValueError( f'Grok requested an unregistered tool: "{name}"!' )
            values = json.loads( tool_call.function.arguments )
            if iscoroutinefunction( function ):
                output = await function( **values )
            else:
                output = await asyncio.to_thread( function, **values )
                if isawaitable( output ):
                    output = await output
            chat.append( tool_result(
                self.serialize_result( output ),
                tool_call_id=tool_call.id,
            ) )


    def serialize_result( self, result: Any ) -> str:
        """Serialize a client-side tool result for xAI.

        Purpose:
            Converts tool output into the string content required by xAI while retaining structured
            JSON whenever possible.

        Args:
            result (Any): Value returned by a registered local tool.

        Returns:
            str: Serialized tool-result content.
        """
        if isinstance( result, str ):
            return result
        return json.dumps( result, default=str )
