'''
  ******************************************************************************************
      Assembly:                minions
      Filename:                init.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="init.py" company="Terry D. Eppler">

	     init.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    init.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def throw_if( name: str, value: object ) -> None:
    """Input guard.

    Purpose:
        Validates that a required argument contains a usable value before the surrounding
        workflow continues.

    Args:
        name (str): Name of the argument being validated.
        value (object): Value supplied to the argument.

    Returns:
        None: This function performs validation and does not return a value.
    """
    if not value:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


class Minion( ABC ):
    """Shared workflow-agent contract.

    Purpose:
        Stores configuration and execution state shared by provider-specific workflow agents.
        Concrete agents retain their provider's native runtime and result types.

    Args:
        name (str): Human-readable name assigned to the workflow agent.
        model (str): Provider model identifier used by the workflow agent.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Any]): Tools created for the workflow agent's provider.
        max_turns (int): Maximum number of model turns allowed for one execution.
    """

    name: str
    model: str
    instructions: str
    tools: list[ Any ]
    max_turns: int
    provider: Any
    result: Any
    prompt: str


    def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ],
            max_turns: int=10 ) -> None:
        """Initialize shared workflow-agent state.

        Purpose:
            Validates and stores the configuration used by a provider-specific workflow agent.

        Args:
            name (str): Human-readable name assigned to the workflow agent.
            model (str): Provider model identifier used by the workflow agent.
            instructions (str): System-level instructions controlling agent behavior.
            tools (list[Any]): Tools created for the workflow agent's provider.
            max_turns (int): Maximum number of model turns allowed for one execution.

        Returns:
            None: Initialization stores the validated workflow-agent configuration.
        """
        throw_if( 'name', name )
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if( 'tools', tools )
        throw_if( 'max_turns', max_turns )

        if max_turns < 1:
            raise ValueError( 'Argument "max_turns" must be greater than zero!' )

        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.max_turns = max_turns
        self.provider = None
        self.result = None
        self.prompt = ''


    @abstractmethod
    def create( self ) -> Any:
        """Create the provider runtime.

        Purpose:
            Creates the provider-native agent used by the workflow implementation.

        Returns:
            Any: Provider-native agent instance.
        """
        raise NotImplementedError


    @abstractmethod
    def run( self, prompt: str ) -> Any:
        """Run the workflow agent synchronously.

        Purpose:
            Executes the provider-native agent and retains its native result.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Provider-native execution result.
        """
        raise NotImplementedError


    @abstractmethod
    async def run_async( self, prompt: str ) -> Any:
        """Run the workflow agent asynchronously.

        Purpose:
            Executes the provider-native agent asynchronously and retains its native result.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Provider-native execution result.
        """
        raise NotImplementedError


    @abstractmethod
    def stream( self, prompt: str ) -> Any:
        """Start a streamed workflow-agent execution.

        Purpose:
            Starts the provider-native streaming workflow and retains its streaming result.

        Args:
            prompt (str): User input supplied to the workflow agent.

        Returns:
            Any: Provider-native streaming result.
        """
        raise NotImplementedError


    def reset( self ) -> None:
        """Reset transient execution state.

        Purpose:
            Clears the most recent prompt and result while retaining agent configuration and the
            provider runtime.

        Returns:
            None: Transient state is reset in place.
        """
        self.result = None
        self.prompt = ''


__all__: list[ str ] = [
    'Minion',
    'throw_if',
]
