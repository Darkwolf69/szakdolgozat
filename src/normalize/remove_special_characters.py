# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:12:56 2023

@author: adamWolf
"""

import re


def remove_special_characters(text, remove_digits=False):
    """
    Remove special characters from a text string which are not
    ASCII characters (whitelist)

    Parameters:
        text (str): The input text string from which special characters will be removed.
        remove_digits (bool): Whether to remove digits as well (default is False).

    Returns:
        str: The text string with special characters removed.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
    
    return text