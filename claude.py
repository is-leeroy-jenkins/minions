'''Anthropic Claude Minion implementations.'''
from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar
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

    minion_name: ClassVar[ str ] = 'General Minion'

    def __init__( self, model: str, instructions: str,
            tools: Sequence[ BetaFunctionTool ] | None=None, max_turns: int=10,
            max_tokens: int=4096, api_key: str | None=None,
            name: str | None=None ) -> None:
        """Initialize a Claude Minion.

        Args:
            model (str): Claude model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[BetaFunctionTool] | None): Optional Anthropic beta tools.
            max_turns (int): Maximum model iterations per execution.
            max_tokens (int): Maximum output tokens per model iteration.
            api_key (str | None): Optional Anthropic API key override.
            name (str | None): Optional name overriding the implementation default.

        Returns:
            None: Configuration and native clients are stored.
        """
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        throw_if_less_than( 'max_tokens', max_tokens, 1 )
        self.name = name or self.minion_name
        throw_if( 'name', self.name )
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

    minion_name: ClassVar[ str ] = 'Data Minion'


class ResearchMinion( Minion ):
    """Claude Minion specialized for research workflows."""

    minion_name: ClassVar[ str ] = 'Research Minion'


class CodingMinion( Minion ):
    """Claude Minion specialized for software workflows."""

    minion_name: ClassVar[ str ] = 'Coding Minion'


class WritingMinion( Minion ):
    """Claude Minion specialized for writing workflows."""

    minion_name: ClassVar[ str ] = 'Writing Minion'


class PlanningMinion( Minion ):
    """Claude Minion specialized for planning workflows."""

    minion_name: ClassVar[ str ] = 'Planning Minion'


class ComplianceMinion( Minion ):
    """Claude Minion specialized for compliance, legal, and budget workflows."""

    minion_name: ClassVar[ str ] = 'Compliance Minion'


class BusinessMinion( Minion ):
    """Claude Minion specialized for business, finance, and marketing workflows."""

    minion_name: ClassVar[ str ] = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """Claude Minion specialized for image-generation workflows."""

    minion_name: ClassVar[ str ] = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """Claude Minion specialized for image-analysis workflows."""

    minion_name: ClassVar[ str ] = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """Claude Minion specialized for image-editing workflows."""

    minion_name: ClassVar[ str ] = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """Claude Minion specialized for translation workflows."""

    minion_name: ClassVar[ str ] = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """Claude Minion specialized for transcription workflows."""

    minion_name: ClassVar[ str ] = 'Transcription Minion'


class SpeechMinion( Minion ):
    """Claude Minion specialized for speech workflows."""

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
    'TranscriptionMinion',
    'TranslationMinion',
    'WritingMinion',
]
