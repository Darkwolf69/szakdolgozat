# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:38:56 2023

@author: adamWolf
'''

def prepare_spanish(text):
    """
    Prepares Spanish text for analysis by removing unnecessary em-dash characters.

    Parameters:
        text (str): The Spanish text to be prepared.

    Returns:
        str: The preprocessed Spanish text.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        #remove unnecessary em-dash characters
        prepared_text = text.replace('—', '')
        
    return prepared_text
