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
    Implements data-oriented workflow agents using the Google Agent Development Kit.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4
import re

from google.adk import Agent, Runner
from google.adk.agents import RunConfig
from google.adk.sessions import InMemorySessionService
from google.genai import types

from minions import Minion, throw_if


class DataAgent( Minion ):
    """Google Gemini data workflow agent.

    Purpose:
        Creates and executes a data-oriented workflow agent with Google ADK. Callable tools,
        including operations from ``fonky.gemini.tools``, are registered directly with the ADK
        agent and executed by the provider runtime.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): Gemini model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Callable[..., Any]]): Google ADK-compatible callable tools.
        max_turns (int): Maximum number of model calls allowed for one execution.
    """


    def __init__( self, name: str, model: str, instructions: str,
            tools: list[ Callable[ ..., Any ] ], max_turns: int=10 ) -> None:
        """Initialize the Google Gemini data workflow agent.

        Purpose:
            Validates and stores workflow configuration before creating the provider-native ADK
            agent and runner.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): Gemini model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Callable[..., Any]]): Google ADK-compatible callable tools.
            max_turns (int): Maximum number of model calls allowed for one execution.

        Returns:
            None: Initialization creates and stores the Google ADK runtime.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )

        if not all( callable( tool ) for tool in self.tools ):
            raise ValueError( 'Argument "tools" must contain only callable Gemini tools!' )

        self.provider_name = self.normalize_name( self.name )
        self.app_name = f'minions_{self.provider_name}'
        self.user_id = self.provider_name
        self.run_config = RunConfig( max_llm_calls=self.max_turns )
        self.agent: Agent | None = None
        self.events: list[ Any ] = [ ]
        self.provider = self.create( )


    def normalize_name( self, name: str ) -> str:
        """Normalize a human-readable name for Google ADK.

        Purpose:
            Converts the shared human-readable agent name into the identifier format required by
            Google ADK while preserving the original value in ``self.name``.

        Args:
            name (str): Human-readable agent name.

        Returns:
            str: Valid Google ADK agent identifier.
        """
        throw_if( 'name', name )
        value = re.sub( r'[^A-Za-z0-9_]+', '_', name.strip( ) ).strip( '_' ).lower( )
        if not value:
            raise ValueError( 'Argument "name" must contain letters or numbers!' )
        if value[ 0 ].isdigit( ):
            value = f'agent_{value}'
        return value


    def create( self ) -> Runner:
        """Create the Google ADK agent and runner.

        Purpose:
            Registers provider-compatible tools with a Gemini agent and creates an in-memory ADK
            runner that automatically provisions an isolated session for each execution.

        Returns:
            Runner: Configured Google ADK runner.
        """
        self.agent = Agent(
            name=self.provider_name,
            model=self.model,
            instruction=self.instructions,
            tools=self.tools,
        )
        self.provider = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=InMemorySessionService( ),
            auto_create_session=True,
        )
        return self.provider


    def create_content( self, prompt: str ) -> Any:
        """Create provider-native Gemini user content.

        Purpose:
            Converts validated prompt text into the Google Gen AI content object required by the
            ADK runner.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Google Gen AI ``Content`` instance.
        """
        throw_if( 'prompt', prompt )
        return types.Content(
            role='user',
            parts=[ types.Part.from_text( text=prompt ) ],
        )


    def create_session_id( self ) -> str:
        """Create an isolated Google ADK session identifier.

        Purpose:
            Creates a unique session for each top-level workflow execution so prompts do not
            unintentionally share provider conversation state.

        Returns:
            str: Unique session identifier.
        """
        return uuid4( ).hex


    def select_result( self, events: list[ Any ] ) -> Any:
        """Select the final provider event from an ADK invocation.

        Purpose:
            Retains the final-response event when available and otherwise returns the last native
            event emitted by the runner.

        Args:
            events (list[Any]): Provider-native ADK events emitted during execution.

        Returns:
            Any: Final provider-native ADK event.
        """
        throw_if( 'events', events )
        final_events = [ event for event in events
            if callable( getattr( event, 'is_final_response', None ) )
            and event.is_final_response( ) ]
        return final_events[ -1 ] if final_events else events[ -1 ]


    def run( self, prompt: str ) -> Any:
        """Run the Google Gemini data workflow synchronously.

        Purpose:
            Executes a bounded Google ADK invocation, including automatic Fonky callable-tool
            execution, and returns the final provider-native event.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Google ADK event.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.events = list( self.provider.run(
            user_id=self.user_id,
            session_id=self.create_session_id( ),
            new_message=self.create_content( self.prompt ),
            run_config=self.run_config,
        ) )
        self.result = self.select_result( self.events )
        return self.result


    async def run_async( self, prompt: str ) -> Any:
        """Run the Google Gemini data workflow asynchronously.

        Purpose:
            Executes a bounded asynchronous Google ADK invocation, including automatic Fonky
            callable-tool execution, and returns the final provider-native event.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Final Google ADK event.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.events = [ event async for event in self.provider.run_async(
            user_id=self.user_id,
            session_id=self.create_session_id( ),
            new_message=self.create_content( self.prompt ),
            run_config=self.run_config,
        ) ]
        self.result = self.select_result( self.events )
        return self.result


    def stream( self, prompt: str ) -> AsyncIterator[ Any ]:
        """Start a streamed Google Gemini data workflow.

        Purpose:
            Returns the native asynchronous ADK event stream. Google ADK continues to execute
            registered callable tools within the streamed invocation.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            AsyncIterator[Any]: Provider-native Google ADK event stream.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = self.provider.run_async(
            user_id=self.user_id,
            session_id=self.create_session_id( ),
            new_message=self.create_content( self.prompt ),
            run_config=self.run_config,
        )
        return self.result
