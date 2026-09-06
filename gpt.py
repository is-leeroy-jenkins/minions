'''OpenAI Agents SDK Minion implementations.'''
from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from agents import Agent, RunResult, RunResultStreaming, Runner, Tool

from . import throw_if, throw_if_less_than


class Minion( Agent ):
    """Provider-native OpenAI workflow agent."""

    minion_name: ClassVar[ str ] = 'General Minion'
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
        throw_if_less_than( 'max_turns', max_turns, 1 )
        agent_name = name or self.minion_name
        throw_if( 'name', agent_name )
        super( ).__init__(
            name=agent_name,
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

    minion_name: ClassVar[ str ] = 'Data Minion'


class ResearchMinion( Minion ):
    """OpenAI Minion specialized for research workflows."""

    minion_name: ClassVar[ str ] = 'Research Minion'


class CodingMinion( Minion ):
    """OpenAI Minion specialized for software workflows."""

    minion_name: ClassVar[ str ] = 'Coding Minion'


class WritingMinion( Minion ):
    """OpenAI Minion specialized for writing workflows."""

    minion_name: ClassVar[ str ] = 'Writing Minion'


class PlanningMinion( Minion ):
    """OpenAI Minion specialized for planning workflows."""

    minion_name: ClassVar[ str ] = 'Planning Minion'


class ComplianceMinion( Minion ):
    """OpenAI Minion specialized for compliance, legal, and budget workflows."""

    minion_name: ClassVar[ str ] = 'Compliance Minion'


class BusinessMinion( Minion ):
    """OpenAI Minion specialized for business, finance, and marketing workflows."""

    minion_name: ClassVar[ str ] = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """OpenAI Minion specialized for image-generation workflows."""

    minion_name: ClassVar[ str ] = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """OpenAI Minion specialized for image-analysis workflows."""

    minion_name: ClassVar[ str ] = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """OpenAI Minion specialized for image-editing workflows."""

    minion_name: ClassVar[ str ] = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """OpenAI Minion specialized for translation workflows."""

    minion_name: ClassVar[ str ] = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """OpenAI Minion specialized for transcription workflows."""

    minion_name: ClassVar[ str ] = 'Transcription Minion'


class SpeechMinion( Minion ):
    """OpenAI Minion specialized for speech workflows."""

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
