# -*- coding: utf-8 -*-
'''
Created on Wed Nov 15 22:58:32 2023

@author: adamWolf
'''

def prepare_french(text):
    """
    Prepares the French text for analysis by unifying hyphens.

    Parameters:
        text (str): The French text to be prepared.

    Returns:
        str: The preprocessed French text.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        # unify hyphens (replace EM-DASH-es with HYPHEN-MINUS-es)
        prepared_text = text.replace('—', '-', text)
    
    return(prepared_text)

