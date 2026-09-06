'''Mistral AI Minion implementations.'''
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence
from inspect import isawaitable, iscoroutinefunction
from typing import TypedDict, cast
import asyncio
import json
import os

from mistralai.client import Mistral
from mistralai.client.models import (
    Agent,
    AssistantMessage,
    ChatCompletionResponse,
    CompletionChunk,
    CompletionEvent,
    CreateAgentRequestTool,
    CreateAgentRequestToolTypedDict,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from mistralai.extra.observability.streaming import accumulate_chunks_to_response_dict

from . import throw_if, throw_if_less_than


MistralTool = CreateAgentRequestTool | CreateAgentRequestToolTypedDict
ToolFunction = Callable[ ..., object | Awaitable[ object ] ]
MistralMessage = UserMessage | AssistantMessage | ToolMessage


class AggregatedChoice( TypedDict ):
    '''Relevant choice data returned by the Mistral stream accumulator.'''

    message: dict[ str, object ]


class AggregatedResponse( TypedDict ):
    '''Relevant response data returned by the Mistral stream accumulator.'''

    choices: list[ AggregatedChoice ]


class Minion:
    """Mistral workflow agent with complete local tool execution."""


    def __init__( self, name: str, model: str, instructions: str,
            tools: Sequence[ MistralTool ] | None=None,
            functions: Sequence[ ToolFunction ] | None=None, max_turns: int=10,
            api_key: str | None=None ) -> None:
        """Initialize a Mistral Minion and its remote agent resource.

        Args:
            name (str): Human-readable agent name.
            model (str): Mistral model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[MistralTool] | None): Optional Mistral tool schemas.
            functions (Sequence[ToolFunction] | None): Optional matching local callables.
            max_turns (int): Maximum model turns per execution.
            api_key (str | None): Optional Mistral API key override.

        Returns:
            None: Configuration, client, and remote agent are stored.
        """
        throw_if( 'name', name )
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = list( tools or [ ] )
        function_list = list( functions or [ ] )
        self.functions = { function.__name__: function for function in function_list }
        if len( self.functions ) != len( function_list ):
            raise ValueError( 'Argument "functions" must contain unique names!' )
        self.validate_tools( )
        self.max_turns = max_turns
        self.api_key = api_key or os.getenv( 'MISTRAL_API_KEY' )
        throw_if( 'api_key', self.api_key )
        self.client = Mistral( api_key=self.api_key )
        self.provider = self.create( )
        self.result: ChatCompletionResponse | AssistantMessage | None = None


    def get_tool_name( self, tool: MistralTool ) -> str | None:
        """Read a local Mistral function-tool name when present.

        Args:
            tool (MistralTool): Native model or typed dictionary schema.

        Returns:
            str | None: Declared function name, or None for provider-hosted tools.
        """
        if isinstance( tool, dict ):
            if tool.get( 'type' ) != 'function':
                return None
            function = tool.get( 'function' )
            if not isinstance( function, dict ):
                raise TypeError( 'Mistral function tool requires a function object!' )
            name = function.get( 'name' )
        else:
            function = getattr( tool, 'function', None )
            if function is None:
                return None
            name = function.name
        throw_if( 'tool.function.name', name )
        return str( name )


    def validate_tools( self ) -> None:
        """Validate exact schema-to-callable pairing when tools are supplied.

        Returns:
            None: Valid pairs are retained on the Minion.
        """
        names = [ name for tool in self.tools
            if ( name := self.get_tool_name( tool ) ) is not None ]
        if len( names ) != len( set( names ) ):
            raise ValueError( 'Argument "tools" must contain unique function names!' )
        missing = sorted( set( names ).difference( self.functions ) )
        extra = sorted( set( self.functions ).difference( names ) )
        if missing or extra:
            raise ValueError(
                f'Mistral tools and functions must match exactly; missing={missing}, extra={extra}!'
            )


    def create( self ) -> Agent:
        """Create the provider-native remote Mistral agent.

        Returns:
            Agent: Mistral agent resource used for executions.
        """
        return self.client.beta.agents.create(
            name=self.name,
            model=self.model,
            instructions=self.instructions,
            tools=self.tools or None,
        )


    def run( self, prompt: str ) -> ChatCompletionResponse:
        """Execute the complete Mistral workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            ChatCompletionResponse: Final provider-native completion.
        """
        throw_if( 'prompt', prompt )
        messages: list[ MistralMessage ] = [ UserMessage( content=prompt ) ]
        for _ in range( self.max_turns ):
            result = self.client.agents.complete(
                agent_id=self.provider.id, messages=messages
            )
            self.result = result
            message = result.choices[ 0 ].message
            tool_calls = message.tool_calls or [ ]
            if not tool_calls:
                return result
            messages.append( message )
            messages.extend( self.execute_tools( tool_calls ) )
        raise RuntimeError( f'Mistral workflow exceeded {self.max_turns} model turns!' )


    async def run_async( self, prompt: str ) -> ChatCompletionResponse:
        """Execute the complete Mistral workflow asynchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            ChatCompletionResponse: Final provider-native completion.
        """
        throw_if( 'prompt', prompt )
        messages: list[ MistralMessage ] = [ UserMessage( content=prompt ) ]
        for _ in range( self.max_turns ):
            result = await self.client.agents.complete_async(
                agent_id=self.provider.id, messages=messages
            )
            self.result = result
            message = result.choices[ 0 ].message
            tool_calls = message.tool_calls or [ ]
            if not tool_calls:
                return result
            messages.append( message )
            messages.extend( await self.execute_tools_async( tool_calls ) )
        raise RuntimeError( f'Mistral workflow exceeded {self.max_turns} model turns!' )


    def stream( self, prompt: str ) -> Iterator[ CompletionEvent ]:
        """Start a streamed workflow that continues across local tool calls.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            Iterator[CompletionEvent]: Provider events from every model turn.
        """
        throw_if( 'prompt', prompt )
        return self.stream_events( prompt )


    def stream_events( self, prompt: str ) -> Iterator[ CompletionEvent ]:
        """Execute and stream the complete Mistral tool loop.

        Args:
            prompt (str): Validated user input.

        Yields:
            CompletionEvent: Native event from the active Mistral stream.
        """
        messages: list[ MistralMessage ] = [ UserMessage( content=prompt ) ]
        for _ in range( self.max_turns ):
            chunks: list[ CompletionChunk ] = [ ]
            for event in self.client.agents.stream(
                    agent_id=self.provider.id, messages=messages ):
                chunks.append( event.data )
                yield event
            if not chunks:
                raise RuntimeError( 'Mistral streaming returned no completion chunks!' )
            aggregate = cast(
                AggregatedResponse, accumulate_chunks_to_response_dict( chunks )
            )
            message = AssistantMessage.model_validate(
                aggregate[ 'choices' ][ 0 ][ 'message' ]
            )
            self.result = message
            tool_calls = message.tool_calls or [ ]
            if not tool_calls:
                return
            messages.append( message )
            messages.extend( self.execute_tools( tool_calls ) )
        raise RuntimeError( f'Mistral workflow exceeded {self.max_turns} model turns!' )


    def execute_tools( self, tool_calls: Sequence[ ToolCall ] ) -> list[ ToolMessage ]:
        """Execute synchronous Mistral tool calls.

        Args:
            tool_calls (Sequence[ToolCall]): Provider-requested function calls.

        Returns:
            list[ToolMessage]: Correlated tool-result messages.
        """
        messages: list[ ToolMessage ] = [ ]
        for tool_call in tool_calls:
            function = self.functions.get( tool_call.function.name )
            if function is None:
                raise ValueError(
                    f'Mistral requested an unregistered tool: "{tool_call.function.name}"!'
                )
            arguments = self.parse_arguments( tool_call.function.arguments )
            output = function( **arguments )
            if isawaitable( output ):
                raise TypeError( f'Tool "{tool_call.function.name}" requires async execution!' )
            messages.append( ToolMessage(
                name=tool_call.function.name,
                content=self.serialize_result( output ),
                tool_call_id=tool_call.id,
            ) )
        return messages


    async def execute_tools_async( self,
            tool_calls: Sequence[ ToolCall ] ) -> list[ ToolMessage ]:
        """Execute asynchronous Mistral tool calls.

        Args:
            tool_calls (Sequence[ToolCall]): Provider-requested function calls.

        Returns:
            list[ToolMessage]: Correlated tool-result messages.
        """
        messages: list[ ToolMessage ] = [ ]
        for tool_call in tool_calls:
            function = self.functions.get( tool_call.function.name )
            if function is None:
                raise ValueError(
                    f'Mistral requested an unregistered tool: "{tool_call.function.name}"!'
                )
            arguments = self.parse_arguments( tool_call.function.arguments )
            if iscoroutinefunction( function ):
                output = await function( **arguments )
            else:
                output = await asyncio.to_thread( function, **arguments )
                if isawaitable( output ):
                    output = await output
            messages.append( ToolMessage(
                name=tool_call.function.name,
                content=self.serialize_result( output ),
                tool_call_id=tool_call.id,
            ) )
        return messages


    @staticmethod
    def parse_arguments( arguments: str | dict[ str, object ] ) -> dict[ str, object ]:
        """Parse and validate model-supplied tool arguments.

        Args:
            arguments (str | dict[str, object]): JSON or decoded keyword arguments.

        Returns:
            dict[str, object]: Validated keyword arguments.
        """
        values = json.loads( arguments ) if isinstance( arguments, str ) else arguments
        if not isinstance( values, dict ):
            raise TypeError( 'Tool arguments must be a JSON object!' )
        return values


    @staticmethod
    def serialize_result( result: object ) -> str:
        """Serialize a local tool result.

        Args:
            result (object): Value returned by a tool.

        Returns:
            str: Provider-compatible result content.
        """
        return result if isinstance( result, str ) else json.dumps( result, default=str )


class DataMinion( Minion ):
    """Mistral Minion specialized for data workflows."""


    def __init__( self, model: str, instructions: str,
            tools: Sequence[ MistralTool ] | None=None,
            functions: Sequence[ ToolFunction ] | None=None, max_turns: int=10,
            api_key: str | None=None, name: str='Data Minion' ) -> None:
        """Initialize a Mistral data Minion.

        Args:
            model (str): Mistral model identifier.
            instructions (str): Data-workflow instructions.
            tools (Sequence[MistralTool] | None): Optional Mistral tool schemas.
            functions (Sequence[ToolFunction] | None): Optional matching callables.
            max_turns (int): Maximum model turns per execution.
            api_key (str | None): Optional Mistral API key override.
            name (str): Human-readable agent name.

        Returns:
            None: Configuration, client, and remote agent are stored.
        """
        super( ).__init__(
            name, model, instructions, tools, functions, max_turns, api_key
        )


__all__: list[ str ] = [ 'DataMinion', 'Minion', 'MistralTool', 'ToolFunction' ]
