'''Public package utilities for provider-specific Minions.'''
from __future__ import annotations


def throw_if( name: str, value: object ) -> None:
    """Validate a required value.

    Purpose:
        Raises a consistent error when a required value is empty.

    Args:
        name (str): Argument name included in the error.
        value (object): Value to validate.

    Returns:
        None: Validation succeeds without returning a value.
    """
    if not value:
        raise ValueError( f'Argument "{name}" cannot be empty!' )


def throw_if_less_than( name: str, value: int, minimum: int ) -> None:
    """Validate a numeric lower bound.

    Args:
        name (str): Argument name included in the error.
        value (int): Integer to validate.
        minimum (int): Smallest accepted value.

    Returns:
        None: Validation succeeds without returning a value.
    """
    if value < minimum:
        raise ValueError( f'Argument "{name}" must be at least {minimum}!' )


__version__ = '0.2.0'

__all__: list[ str ] = [ 'throw_if', 'throw_if_less_than' ]
