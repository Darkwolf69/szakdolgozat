# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:52:32 2023

@author: adamWolf

works only with less than a specific length of string
"""

import contractions


def expand_contractions(text):
    """
    Expands contractions in the given text.

    Parameters:
        text (str): The input text containing contractions to be expanded.

    Returns:
        str: The text with contractions expanded.

    Raises:
        AttributeError: If the input is not a string.
        
    
    Examples:
        >>> expand_contractions("I'm going to the store. She's not here.")
        'I am going to the store. She is not here.'

        >>> expand_contractions("He'll be here soon. They're coming later.")
        'He will be here soon. They are coming later.'

        >>> expand_contractions(12345)
        Traceback (most recent call last):
        AttributeError: Invalid input

    """
    
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        return contractions.fix(text)

if __name__ == "__main__":
    import doctest
    doctest.testmod()