'''xAI Grok Minion implementations.'''
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from inspect import isawaitable, iscoroutinefunction
from typing import ClassVar, overload
import asyncio
import json
import os

from xai_sdk import AsyncClient, Client
from xai_sdk.aio.chat import Chat as AsyncChat
from xai_sdk.chat import Chunk, Response, system, tool_result, user
from xai_sdk.proto import chat_pb2
from xai_sdk.sync.chat import Chat as SyncChat

from . import throw_if, throw_if_less_than


ToolFunction = Callable[ ..., object | Awaitable[ object ] ]


class Minion:
    """Grok workflow agent backed by xAI client-side tool calling."""

    minion_name: ClassVar[ str ] = 'General Minion'

    def __init__( self, model: str, instructions: str,
            tools: Sequence[ chat_pb2.Tool ] | None=None,
            functions: Sequence[ ToolFunction ] | None=None, max_turns: int=10,
            api_key: str | None=None, name: str | None=None ) -> None:
        """Initialize a Grok Minion.

        Args:
            model (str): Grok model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[chat_pb2.Tool] | None): Optional xAI tool schemas.
            functions (Sequence[ToolFunction] | None): Optional matching tool callables.
            max_turns (int): Maximum model turns per execution.
            api_key (str | None): Optional xAI API key override.
            name (str | None): Optional name overriding the implementation default.

        Returns:
            None: Configuration and native clients are stored.
        """
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        self.name = name or self.minion_name
        throw_if( 'name', self.name )
        self.model = model
        self.instructions = instructions
        self.tools = list( tools or [ ] )
        function_list = list( functions or [ ] )
        self.functions = { function.__name__: function for function in function_list }
        if len( self.functions ) != len( function_list ):
            raise ValueError( 'Argument "functions" must contain unique names!' )
        self.validate_tools( )
        self.max_turns = max_turns
        self.api_key = api_key or os.getenv( 'XAI_API_KEY' )
        throw_if( 'api_key', self.api_key )
        self.client = Client( api_key=self.api_key )
        self.async_client = AsyncClient( api_key=self.api_key )
        self.result: Response | None = None


    def validate_tools( self ) -> None:
        """Validate exact schema-to-callable pairing when tools are supplied.

        Returns:
            None: Valid pairs are retained on the Minion.
        """
        names = [ tool.function.name for tool in self.tools ]
        if len( names ) != len( set( names ) ):
            raise ValueError( 'Argument "tools" must contain unique function names!' )
        missing = sorted( set( names ).difference( self.functions ) )
        extra = sorted( set( self.functions ).difference( names ) )
        if missing or extra:
            raise ValueError(
                f'Grok tools and functions must match exactly; missing={missing}, extra={extra}!'
            )


    @overload
    def create_chat( self, client: Client ) -> SyncChat: ...


    @overload
    def create_chat( self, client: AsyncClient ) -> AsyncChat: ...


    def create_chat( self, client: Client | AsyncClient ) -> SyncChat | AsyncChat:
        """Create an isolated provider-native chat.

        Args:
            client (Client | AsyncClient): xAI client used to create the chat.

        Returns:
            SyncChat | AsyncChat: Provider-native chat instance.
        """
        return client.chat.create(
            model=self.model,
            messages=[ system( self.instructions ) ],
            tools=self.tools or None,
        )


    def run( self, prompt: str ) -> Response:
        """Execute the complete Grok workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            Response: Final provider-native response.
        """
        throw_if( 'prompt', prompt )
        chat = self.create_chat( self.client )
        chat.append( user( prompt ) )
        for _ in range( self.max_turns ):
            response = chat.sample( )
            chat.append( response )
            self.result = response
            if not response.tool_calls:
                return response
            self.execute_tools( chat, response.tool_calls )
        raise RuntimeError( f'Grok workflow exceeded {self.max_turns} model turns!' )


    async def run_async( self, prompt: str ) -> Response:
        """Execute the complete Grok workflow asynchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            Response: Final provider-native response.
        """
        throw_if( 'prompt', prompt )
        chat = self.create_chat( self.async_client )
        chat.append( user( prompt ) )
        for _ in range( self.max_turns ):
            response = await chat.sample( )
            chat.append( response )
            self.result = response
            if not response.tool_calls:
                return response
            await self.execute_tools_async( chat, response.tool_calls )
        raise RuntimeError( f'Grok workflow exceeded {self.max_turns} model turns!' )


    async def stream( self, prompt: str ) -> AsyncIterator[ tuple[ Response, Chunk ] ]:
        """Stream the complete Grok workflow across tool turns.

        Args:
            prompt (str): User input for the workflow.

        Yields:
            tuple[Response, Chunk]: Provider-native response and incremental chunk.
        """
        throw_if( 'prompt', prompt )
        chat = self.create_chat( self.async_client )
        chat.append( user( prompt ) )
        for _ in range( self.max_turns ):
            response: Response | None = None
            async for response, chunk in chat.stream( ):
                self.result = response
                yield response, chunk
            if response is None:
                raise RuntimeError( 'Grok streaming returned no response!' )
            chat.append( response )
            if not response.tool_calls:
                return
            await self.execute_tools_async( chat, response.tool_calls )
        raise RuntimeError( f'Grok workflow exceeded {self.max_turns} model turns!' )


    def execute_tools( self, chat: SyncChat,
            tool_calls: Sequence[ chat_pb2.ToolCall ] ) -> None:
        """Execute and append synchronous tool results.

        Args:
            chat (SyncChat): Native chat receiving tool results.
            tool_calls (Sequence[chat_pb2.ToolCall]): Requested xAI tool calls.

        Returns:
            None: Correlated results are appended to the chat.
        """
        for tool_call in tool_calls:
            function = self.functions.get( tool_call.function.name )
            if function is None:
                raise ValueError(
                    f'Grok requested an unregistered tool: "{tool_call.function.name}"!'
                )
            arguments = self.parse_arguments( tool_call.function.arguments )
            output = function( **arguments )
            if isawaitable( output ):
                raise TypeError( f'Tool "{tool_call.function.name}" requires async execution!' )
            chat.append( tool_result(
                self.serialize_result( output ), tool_call_id=tool_call.id
            ) )


    async def execute_tools_async( self, chat: AsyncChat,
            tool_calls: Sequence[ chat_pb2.ToolCall ] ) -> None:
        """Execute and append asynchronous tool results.

        Args:
            chat (AsyncChat): Native async chat receiving tool results.
            tool_calls (Sequence[chat_pb2.ToolCall]): Requested xAI tool calls.

        Returns:
            None: Correlated results are appended to the chat.
        """
        for tool_call in tool_calls:
            function = self.functions.get( tool_call.function.name )
            if function is None:
                raise ValueError(
                    f'Grok requested an unregistered tool: "{tool_call.function.name}"!'
                )
            arguments = self.parse_arguments( tool_call.function.arguments )
            if iscoroutinefunction( function ):
                output = await function( **arguments )
            else:
                output = await asyncio.to_thread( function, **arguments )
                if isawaitable( output ):
                    output = await output
            chat.append( tool_result(
                self.serialize_result( output ), tool_call_id=tool_call.id
            ) )


    @staticmethod
    def parse_arguments( arguments: str ) -> dict[ str, object ]:
        """Parse and validate model-supplied JSON arguments.

        Args:
            arguments (str): JSON object supplied by Grok.

        Returns:
            dict[str, object]: Validated keyword arguments.
        """
        values = json.loads( arguments )
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
    """Grok Minion specialized for data workflows."""

    minion_name: ClassVar[ str ] = 'Data Minion'


class ResearchMinion( Minion ):
    """Grok Minion specialized for research workflows."""

    minion_name: ClassVar[ str ] = 'Research Minion'


class CodingMinion( Minion ):
    """Grok Minion specialized for software workflows."""

    minion_name: ClassVar[ str ] = 'Coding Minion'


class WritingMinion( Minion ):
    """Grok Minion specialized for writing workflows."""

    minion_name: ClassVar[ str ] = 'Writing Minion'


class PlanningMinion( Minion ):
    """Grok Minion specialized for planning workflows."""

    minion_name: ClassVar[ str ] = 'Planning Minion'


class ComplianceMinion( Minion ):
    """Grok Minion specialized for compliance, legal, and budget workflows."""

    minion_name: ClassVar[ str ] = 'Compliance Minion'


class BusinessMinion( Minion ):
    """Grok Minion specialized for business, finance, and marketing workflows."""

    minion_name: ClassVar[ str ] = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """Grok Minion specialized for image-generation workflows."""

    minion_name: ClassVar[ str ] = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """Grok Minion specialized for image-analysis workflows."""

    minion_name: ClassVar[ str ] = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """Grok Minion specialized for image-editing workflows."""

    minion_name: ClassVar[ str ] = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """Grok Minion specialized for translation workflows."""

    minion_name: ClassVar[ str ] = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """Grok Minion specialized for transcription workflows."""

    minion_name: ClassVar[ str ] = 'Transcription Minion'


class SpeechMinion( Minion ):
    """Grok Minion specialized for speech workflows."""

    minion_name: ClassVar[ str ] = 'Speech Minion'


__all__: list[ str ] = [
    'BusinessMinion',
    'CodingMinion',
    'ComplianceMinion',
    'DataMinion',
    'ImageAnalysisMinion',
    'ImageEditingMinion',
    'ImageGenerationMinion',
    'Minion',
    'PlanningMinion',
    'ResearchMinion',
    'SpeechMinion',
    'ToolFunction',
    'TranscriptionMinion',
    'TranslationMinion',
    'WritingMinion',
]
