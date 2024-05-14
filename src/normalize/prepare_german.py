# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:02:34 2023

@author: adamWolf
'''

import re


def prepare_german(text):
    """
    Prepares German text for analysis by removing unnecessary footer texts and page numbers.

    Parameters:
        text (str): The German text to be prepared.

    Returns:
        str: The preprocessed German text.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        #remove unnecessary footer texts and page numbers
        prepared_text = re.sub('HARRY POTTER [a-z].*', '', text)
    
    return(prepared_text)