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
    Implements data-oriented workflow agents using the OpenAI Agents SDK.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any

from agents import Agent, Runner

from minions import Minion, throw_if


class DataAgent( Minion ):
    """OpenAI data workflow agent.

    Purpose:
        Creates and executes a data-oriented workflow agent with the OpenAI Agents SDK. Tools
        supplied to this class must be compatible with the OpenAI provider, including tools from
        ``fonky.gpt.tools``.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): OpenAI model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Any]): OpenAI-compatible tools exposed to the workflow agent.
        max_turns (int): Maximum number of model turns allowed for one execution.
    """


    def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ],
            max_turns: int=10 ) -> None:
        """Initialize the OpenAI data workflow agent.

        Purpose:
            Validates and stores workflow configuration before creating the provider-native agent.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): OpenAI model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Any]): OpenAI-compatible tools exposed to the workflow agent.
            max_turns (int): Maximum number of model turns allowed for one execution.

        Returns:
            None: Initialization creates and stores the OpenAI agent.
        """
        super( ).__init__( name, model, instructions, tools, max_turns )
        self.provider = self.create( )


    def create( self ) -> Agent:
        """Create the OpenAI agent.

        Purpose:
            Creates an OpenAI Agents SDK agent from the validated workflow configuration.

        Returns:
            Agent: Configured OpenAI Agents SDK agent.
        """
        self.provider = Agent(
            name=self.name,
            model=self.model,
            instructions=self.instructions,
            tools=self.tools,
        )
        return self.provider


    def run( self, prompt: str ) -> Any:
        """Run the OpenAI data workflow synchronously.

        Purpose:
            Executes the OpenAI data workflow synchronously with the stored provider and turn
            limit.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: OpenAI Agents SDK run result.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = Runner.run_sync(
            starting_agent=self.provider,
            input=self.prompt,
            max_turns=self.max_turns,
        )
        return self.result


    async def run_async( self, prompt: str ) -> Any:
        """Run the OpenAI data workflow asynchronously.

        Purpose:
            Executes the OpenAI data workflow asynchronously with the stored provider and turn
            limit.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: OpenAI Agents SDK run result.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = await Runner.run(
            starting_agent=self.provider,
            input=self.prompt,
            max_turns=self.max_turns,
        )
        return self.result


    def stream( self, prompt: str ) -> Any:
        """Start a streamed OpenAI data workflow.

        Purpose:
            Starts an OpenAI streaming run with the stored provider and turn limit.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: OpenAI Agents SDK streaming result.
        """
        throw_if( 'prompt', prompt )
        self.prompt = prompt
        self.result = Runner.run_streamed(
            starting_agent=self.provider,
            input=self.prompt,
            max_turns=self.max_turns,
        )
        return self.result
