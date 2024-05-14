# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:39:29 2023

@author: adamWolf
"""

def remove_empty_lines(text):
    """
    Removes empty lines from the given text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with empty lines removed.

    Raises:
        AttributeError: If the input is not a string.
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip() != '']
        text = '\n'.join(non_empty_lines)
        
    return text