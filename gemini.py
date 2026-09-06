'''Google Agent Development Kit Minion implementations.'''
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import ClassVar
from uuid import uuid4
import re

from google.adk import Agent, Runner
from google.adk.agents import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types
from pydantic import Field

from . import throw_if, throw_if_less_than


GeminiTool = Callable[ ..., object ] | BaseTool | BaseToolset


class Minion( Agent ):
    """Provider-native Gemini workflow agent."""

    minion_name: ClassVar[ str ] = 'Gemini Minion'
    display_name: str
    max_turns: int = 10
    prompt: str = ''
    result: Event | None = None
    events: list[ Event ] = Field( default_factory=list )


    def __init__( self, model: str, instructions: str,
            tools: Sequence[ GeminiTool ] | None=None, max_turns: int=10,
            name: str | None=None ) -> None:
        """Initialize a Gemini Minion.

        Args:
            model (str): Gemini model identifier.
            instructions (str): System instructions for the agent.
            tools (Sequence[GeminiTool] | None): Optional Google ADK tools.
            max_turns (int): Maximum model calls per execution.
            name (str | None): Optional name overriding the implementation default.

        Returns:
            None: Configuration is stored on the provider-native agent.
        """
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if_less_than( 'max_turns', max_turns, 1 )
        display_name = name or self.minion_name
        throw_if( 'name', display_name )
        provider_name = self.normalize_name( display_name )
        super( ).__init__( name=provider_name, display_name=display_name, model=model,
            instruction=instructions, tools=list( tools or [ ] ), max_turns=max_turns, )


    @staticmethod
    def normalize_name( name: str ) -> str:
        """Create a valid Google ADK agent identifier.

        Args:
            name (str): Human-readable agent name.

        Returns:
            str: Normalized ADK identifier.
        """
        value = re.sub( r'[^A-Za-z0-9_]+', '_', name.strip( ) ).strip( '_' ).lower( )
        throw_if( 'name', value )
        return f'agent_{value}' if value[ 0 ].isdigit( ) else value


    def create_runner( self ) -> Runner:
        """Create an isolated provider runner.

        Returns:
            Runner: Google ADK runner with automatic session creation.
        """
        return Runner(
            agent=self,
            app_name=f'minions_{self.name}',
            session_service=InMemorySessionService( ),
            auto_create_session=True,
        )


    def create_content( self, prompt: str ) -> types.Content:
        """Create provider-native user content.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            types.Content: Google Gen AI content object.
        """
        throw_if( 'prompt', prompt )
        return types.Content( role='user', parts=[ types.Part.from_text( text=prompt ) ] )


    def select_result( self, events: Sequence[ Event ] ) -> Event:
        """Select the final event from an invocation.

        Args:
            events (Sequence[Event]): Events emitted by Google ADK.

        Returns:
            Event: Final-response event or the last emitted event.
        """
        throw_if( 'events', events )
        final_events = [ event for event in events if event.is_final_response( ) ]
        return final_events[ -1 ] if final_events else events[ -1 ]


    def run( self, prompt: str ) -> Event:
        """Execute the complete Gemini workflow synchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            Event: Final provider-native event.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        runner = self.create_runner( )
        self.events = list( runner.run( user_id=self.name, session_id=uuid4( ).hex,
            new_message=self.create_content( prompt ),
            run_config=RunConfig( max_llm_calls=self.max_turns ), ) )
        self.result = self.select_result( self.events )
        return self.result


    async def run_async( self, prompt: str ) -> Event:
        """Execute the complete Gemini workflow asynchronously.

        Args:
            prompt (str): User input for the workflow.

        Returns:
            Event: Final provider-native event.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        runner = self.create_runner( )
        self.events = [ event async for event in
                        runner.run_async( user_id=self.name, session_id=uuid4( ).hex,
                            new_message=self.create_content( prompt ),
                            run_config=RunConfig( max_llm_calls=self.max_turns ), ) ]
        self.result = self.select_result( self.events )
        return self.result


    async def stream( self, prompt: str ) -> AsyncIterator[ Event ]:
        """Stream a complete Gemini workflow with automatic tool handling.

        Args:
            prompt (str): User input for the workflow.

        Yields:
            Event: Provider-native streamed ADK event.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.events = [ ]
        runner = self.create_runner( )
        async for event in runner.run_async( user_id=self.name, session_id=uuid4( ).hex,
                new_message=self.create_content( prompt ),
                run_config=RunConfig( max_llm_calls=self.max_turns,
                    streaming_mode=StreamingMode.SSE, ) ):
            self.events.append( event )
            if event.is_final_response( ):
                self.result = event
            yield event
        if self.events and self.result is None:
            self.result = self.events[ -1 ]


class DataMinion( Minion ):
    """Gemini Minion specialized for data workflows."""

    minion_name: ClassVar[ str ] = 'Data Minion'


class GovernanceMinion( Minion ):
    """Gemini Minion specialized for governance workflows."""

    minion_name: ClassVar[ str ] = 'Governance Minion'


class ResearchMinion( Minion ):
    """Gemini Minion specialized for research workflows."""

    minion_name: ClassVar[ str ] = 'Research Minion'


class CodingMinion( Minion ):
    """Gemini Minion specialized for software workflows."""

    minion_name: ClassVar[ str ] = 'Coding Minion'


class WritingMinion( Minion ):
    """Gemini Minion specialized for writing workflows."""

    minion_name: ClassVar[ str ] = 'Writing Minion'


class PlanningMinion( Minion ):
    """Gemini Minion specialized for planning workflows."""

    minion_name: ClassVar[ str ] = 'Planning Minion'


class ComplianceMinion( Minion ):
    """Gemini Minion specialized for compliance, legal, and budget workflows."""

    minion_name: ClassVar[ str ] = 'Compliance Minion'


class BusinessMinion( Minion ):
    """Gemini Minion specialized for business, finance, and marketing workflows."""

    minion_name: ClassVar[ str ] = 'Business Minion'


class ImageGenerationMinion( Minion ):
    """Gemini Minion specialized for image-generation workflows."""

    minion_name: ClassVar[ str ] = 'Image Generation Minion'


class ImageAnalysisMinion( Minion ):
    """Gemini Minion specialized for image-analysis workflows."""

    minion_name: ClassVar[ str ] = 'Image Analysis Minion'


class ImageEditingMinion( Minion ):
    """Gemini Minion specialized for image-editing workflows."""

    minion_name: ClassVar[ str ] = 'Image Editing Minion'


class TranslationMinion( Minion ):
    """Gemini Minion specialized for translation workflows."""

    minion_name: ClassVar[ str ] = 'Translation Minion'


class TranscriptionMinion( Minion ):
    """Gemini Minion specialized for transcription workflows."""

    minion_name: ClassVar[ str ] = 'Transcription Minion'


class SpeechMinion( Minion ):
    """Gemini Minion specialized for speech workflows."""

    minion_name: ClassVar[ str ] = 'Speech Minion'

__all__: list[ str ] = [ 'BusinessMinion', 'CodingMinion', 'ComplianceMinion', 'DataMinion',
        'GovernanceMinion', 'GeminiTool', 'ImageAnalysisMinion', 'ImageEditingMinion',
        'ImageGenerationMinion', 'Minion', 'PlanningMinion', 'ResearchMinion', 'SpeechMinion',
        'TranscriptionMinion', 'TranslationMinion', 'WritingMinion', ]
