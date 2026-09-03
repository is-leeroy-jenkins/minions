'''Provider-neutral Minion interface.'''from __future__ import annotations

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
    """Provider-neutral AI agent abstraction.

    Purpose:
        Stores the configuration and transient state shared by Fonky's provider-specific
        Minion implementations. Provider subclasses create and execute their native agent or
        agent-loop objects without leaking those SDK details into this common contract.

    Args:
        name (str): Human-readable name assigned to the Minion.
        model (str): Provider model identifier used by the Minion.
        instructions (str): System-level instructions controlling agent behavior.
        tools (list[Any]): Provider-native tools or Fonky function tools exposed to the agent.
        description (str): Human-readable description of the Minion's purpose.
        max_turns (int): Maximum number of model or tool iterations allowed for one run.
    """

    name: str
    model: str
    instructions: str
    tools: list[ Any ]
    description: str
    max_turns: int
    context: dict[ str, Any ]
    provider: Any
    result: Any
    prompt: str


    def __init__( self, name: str, model: str, instructions: str, tools: list[ Any ],
            description: str='', max_turns: int=10 ) -> None:
        """Initialize shared Minion configuration.

        Purpose:
            Validates and stores the provider-neutral configuration used by every concrete
            Minion implementation.

        Args:
            name (str): Human-readable Minion name.
            model (str): Provider model identifier.
            instructions (str): System-level agent instructions.
            tools (list[Any]): Provider-native tools or Fonky function tools.
            description (str): Optional description of the Minion's purpose.
            max_turns (int): Maximum number of agent execution turns.

        Returns:
            None: Initialization stores validated Minion configuration.
        """
        throw_if( 'name', name )
        throw_if( 'model', model )
        throw_if( 'instructions', instructions )
        throw_if( 'tools', tools )

        if max_turns <= 0:
            raise ValueError( 'Argument "max_turns" must be greater than zero!' )

        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.description = description
        self.max_turns = max_turns
        self.context = { }
        self.provider = None
        self.result = None
        self.prompt = ''


    def add_tool( self, tool: Any ) -> None:
        """Add a tool to the Minion.

        Purpose:
            Adds a provider-native or Fonky function tool when it is not already registered.

        Args:
            tool (Any): Tool object to register.

        Returns:
            None: The tool collection is modified in place.
        """
        throw_if( 'tool', tool )

        if tool not in self.tools:
            self.tools.append( tool )


    def remove_tool( self, tool: Any ) -> None:
        """Remove a tool from the Minion.

        Purpose:
            Removes a registered provider-native or Fonky function tool when present.

        Args:
            tool (Any): Tool object to remove.

        Returns:
            None: The tool collection is modified in place.
        """
        throw_if( 'tool', tool )

        if tool in self.tools:
            self.tools.remove( tool )


    def set_context( self, key: str, value: Any ) -> None:
        """Set a Minion context value.

        Purpose:
            Stores provider-neutral runtime state for subsequent agent execution.

        Args:
            key (str): Context key.
            value (Any): Context value associated with the key.

        Returns:
            None: The context collection is modified in place.
        """
        throw_if( 'key', key )
        self.context[ key ] = value


    def get_context( self, key: str ) -> Any:
        """Get a Minion context value.

        Purpose:
            Retrieves a runtime value previously assigned to the Minion context.

        Args:
            key (str): Context key to retrieve.

        Returns:
            Any: Stored context value, or ``None`` when the key does not exist.
        """
        throw_if( 'key', key )
        return self.context.get( key )


    def clear_context( self ) -> None:
        """Clear Minion runtime context.

        Purpose:
            Removes all provider-neutral runtime state associated with the Minion.

        Returns:
            None: The context collection is cleared in place.
        """
        self.context.clear( )


    @abstractmethod
    def create( self ) -> Any:
        """Create the provider-specific agent runtime.

        Purpose:
            Converts the stored provider-neutral configuration into the provider's native
            agent, client, chat, or runner objects.

        Returns:
            Any: Provider-specific agent or runtime object.
        """
        raise NotImplementedError


    @abstractmethod
    def run( self, prompt: str ) -> Any:
        """Execute the Minion synchronously.

        Args:
            prompt (str): User input supplied to the agent.

        Returns:
            Any: Provider-specific execution result.
        """
        raise NotImplementedError


    @abstractmethod
    async def run_async( self, prompt: str ) -> Any:
        """Execute the Minion asynchronously.

        Args:
            prompt (str): User input supplied to the agent.

        Returns:
            Any: Provider-specific execution result.
        """
        raise NotImplementedError


    @abstractmethod
    def stream( self, prompt: str ) -> Any:
        """Stream a Minion execution.

        Args:
            prompt (str): User input supplied to the agent.

        Returns:
            Any: Provider-specific streaming iterator or event stream.
        """
        raise NotImplementedError


    def reset( self ) -> None:
        """Reset transient Minion state.

        Purpose:
            Clears runtime context, prompt, and result data while preserving permanent
            configuration, provider objects, and registered tools.

        Returns:
            None: Transient state is reset in place.
        """
        self.context.clear( )
        self.result = None
        self.prompt = ''


    def to_dict( self ) -> dict[ str, Any ]:
        """Export Minion configuration.

        Purpose:
            Returns a provider-neutral representation suitable for inspection, logging, or
            serialization.

        Returns:
            dict[str, Any]: Provider-neutral Minion configuration.
        """
        return {
            'name': self.name,
            'model': self.model,
            'instructions': self.instructions,
            'description': self.description,
            'max_turns': self.max_turns,
            'tools': self.tools,
            'context': self.context,
        }


__all__: list[ str ] = [
    'Minion',
    'throw_if',
]
