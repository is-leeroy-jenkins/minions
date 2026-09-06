'''
    ******************************************************************************************
      Assembly:                minions
      Filename:                __init__.py
      Author:                  Terry D. Eppler
      Created:                 09-02-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-06-2026
    ******************************************************************************************
    <copyright file="__init__.py" company="Terry D. Eppler">

         __init__.py
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
        Public package utilities for provider-specific Minions.
    </summary>
    ******************************************************************************************
'''
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


__version__ = '0.2.0'

__all__: list[ str ] = [ 'throw_if' ]
