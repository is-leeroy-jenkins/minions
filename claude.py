'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                claude.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="claude.py" company="Terry D. Eppler">

         claude.py
         Copyright © 2026 Terry D. Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
     PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
     CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
     OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

     You can contact me at: terryeppler@gmail.com or eppler.terry@epa.gov

    </copyright>
    <summary>
        Anthropic Claude Minion implementations.
    </summary>
    ******************************************************************************************
'''
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

from . import throw_if


class Minion:
    """Claude workflow agent backed by Anthropic's native tool runner."""

    minion_name: str = 'Claude Minion'

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
        if max_turns < 1:
            raise ValueError( 'Argument "max_turns" must be at least 1!' )
        if max_tokens < 1:
            raise ValueError( 'Argument "max_tokens" must be at least 1!' )
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
        return beta_async_tool( invoke, name=tool.name, description=tool.description,
            input_schema=tool.input_schema, )


    def run( self, prompt: str ) -> BetaMessage:
        """Execute the complete Claude workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            BetaMessage: Final provider-native message.
        """
        throw_if( 'prompt', prompt )
        runner = self.client.beta.messages.tool_runner( model=self.model,
            max_tokens=self.max_tokens, max_iterations=self.max_turns, system=self.instructions,
            tools=self.tools, messages=[ { 'role': 'user', 'content': prompt } ], )
        result = runner.until_done( )
        self.result = result
        return self.result


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

    minion_name: str = 'Data Minion'


class GovernanceMinion( Minion ):
    """Claude Minion specialized for governance workflows."""

    minion_name: str = 'Governance Minion'


class ResearchMinion( Minion ):
    """Claude Minion specialized for research workflows."""

    minion_name: str = 'Research Minion'


class CodingMinion( Minion ):
    """Claude Minion specialized for software workflows."""

    minion_name: str = 'Coding Minion'


class WritingMinion( Minion ):
    """Claude Minion specialized for writing workflows."""

    minion_name: str = 'Writing Minion'


class PlanningMinion( Minion ):
    """Claude Minion specialized for planning workflows."""

    minion_name: str = 'Planning Minion'


class ComplianceMinion( Minion ):
    """Claude Minion specialized for compliance, legal, and budget workflows."""

    minion_name: str = 'Compliance Minion'


class BusinessMinion( Minion ):
    """Claude Minion specialized for business, finance, and marketing workflows."""

    minion_name: str = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """Claude Minion specialized for image-generation workflows."""

    minion_name: str = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """Claude Minion specialized for image-analysis workflows."""

    minion_name: str = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """Claude Minion specialized for image-editing workflows."""

    minion_name: str = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """Claude Minion specialized for translation workflows."""

    minion_name: str = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """Claude Minion specialized for transcription workflows."""

    minion_name: str = 'Transcription Minion'


class SpeechMinion( Minion ):
    """Claude Minion specialized for speech workflows."""

    minion_name: str = 'Speech Minion'

__all__: list[ str ] = [ 'BusinessMinion', 'CodingMinion', 'ComplianceMinion', 'DataMinion',
        'GovernanceMinion', 'ImageAnalysisMinion', 'ImageEditingMinion', 'ImageGenerationMinion',
        'Minion', 'PlanningMinion', 'ResearchMinion', 'SpeechMinion', 'TranscriptionMinion',
        'TranslationMinion', 'WritingMinion', ]
