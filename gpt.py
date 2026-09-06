'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                gpt.py
      Author:                  Terry D. Eppler
      Created:                 09-05-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="gpt.py" company="Terry D. Eppler">

         gpt.py
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
        OpenAI Agents SDK Minion implementations.
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import Sequence
from agents import Agent, RunResult, RunResultStreaming, Runner, Tool

from . import throw_if


class Minion( Agent ):
    """Provider-native OpenAI workflow agent."""

    minion_name: str = 'GPT Minion'
    max_turns: int
    result: RunResult | RunResultStreaming | None


    def __init__( self, model: str, instructions: str,
            tools: Sequence[ Tool ] | None=None, max_turns: int=10,
            name: str | None=None ) -> None:
        """Initialize an OpenAI Minion.

        Args:
            model (str): OpenAI model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[Tool] | None): Optional OpenAI Agents SDK tools.
            max_turns (int): Maximum model turns per execution.
            name (str | None): Optional name overriding the implementation default.

        Returns:
            None: Configuration is stored on the provider-native agent.
        """
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        if max_turns < 1:
            raise ValueError( 'Argument "max_turns" must be at least 1!' )
        agent_name = name or self.minion_name
        throw_if( 'name', agent_name )
        super( ).__init__( name=agent_name, model=model, instructions=instructions,
            tools=list( tools or [ ] ), )
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

    minion_name: str = 'Data Minion'


class GovernanceMinion( Minion ):
    """OpenAI Minion specialized for governance workflows."""

    minion_name: str = 'Governance Minion'


class ResearchMinion( Minion ):
    """OpenAI Minion specialized for research workflows."""

    minion_name: str = 'Research Minion'


class CodingMinion( Minion ):
    """OpenAI Minion specialized for software workflows."""

    minion_name: str = 'Coding Minion'


class WritingMinion( Minion ):
    """OpenAI Minion specialized for writing workflows."""

    minion_name: str = 'Writing Minion'


class PlanningMinion( Minion ):
    """OpenAI Minion specialized for planning workflows."""

    minion_name: str = 'Planning Minion'


class ComplianceMinion( Minion ):
    """OpenAI Minion specialized for compliance, legal, and budget workflows."""

    minion_name: str = 'Compliance Minion'


class BusinessMinion( Minion ):
    """OpenAI Minion specialized for business, finance, and marketing workflows."""

    minion_name: str = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """OpenAI Minion specialized for image-generation workflows."""

    minion_name: str = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """OpenAI Minion specialized for image-analysis workflows."""

    minion_name: str = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """OpenAI Minion specialized for image-editing workflows."""

    minion_name: str = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """OpenAI Minion specialized for translation workflows."""

    minion_name: str = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """OpenAI Minion specialized for transcription workflows."""

    minion_name: str = 'Transcription Minion'


class SpeechMinion( Minion ):
    """OpenAI Minion specialized for speech workflows."""

    minion_name: str = 'Speech Minion'

__all__: list[ str ] = [ 'BusinessMinion', 'CodingMinion', 'ComplianceMinion', 'DataMinion',
        'GovernanceMinion', 'ImageAnalysisMinion', 'ImageEditingMinion', 'ImageGenerationMinion',
        'Minion', 'PlanningMinion', 'ResearchMinion', 'SpeechMinion', 'TranscriptionMinion',
        'TranslationMinion', 'WritingMinion', ]
