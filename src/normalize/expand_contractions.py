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

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        return contractions.fix(text)
