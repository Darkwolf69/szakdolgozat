# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 23:51:01 2023

@author: adamWolf

Prepare Harry Potter english text for additional normalizing
'''

import re


def prepare_english(text, rm_headers, rm_Hagrid_dialect):
    """
    Prepares the English text for analysis by performing preprocessing steps:
        Removing hyphenations and unifying hyphenated words.
        Optionally removing unnecessary header texts and page numbers.
        Optionally replacing Hagrid's dialect with normal words.
        Removing "er" occurrences which may confuse topic models.

    Parameters:
        text (str): The English text to be prepared.
        rm_headers (bool): If True, removes unnecessary header texts and page numbers.
        rm_Hagrid_dialect (bool): If True, replaces Hagrid's dialect with normal words.

    Returns:
        str: The preprocessed English text.

    Raises:
        AttributeError: If the input text is not a string.
        
    
    Examples:
        >>> text = "10 HARRY  POTTER\\n\\nHello world\\nexample.\\nSecond example, an’ yeh’ll be fine."
        >>> prepare_english(text, rm_headers=True, rm_Hagrid_dialect=True)
        '\\n\\nHello world\\nexample.\\nSecond example, and you will be fine.'
        
        >>> text = "Hello world-\\nexample.\\nSecond example, an’ yeh’ll be fine."
        >>> prepare_english(text, rm_headers=False, rm_Hagrid_dialect=True)
        'Hello worldexample\\n.\\nSecond example, and you will be fine.'
        
        >>> text = "10 HARRY  POTTER\\n-\\nHello world-\\nexample."
        >>> prepare_english(text, rm_headers=True, rm_Hagrid_dialect=False)
        '\\nHello \\nworldexample\\n.'
        
        >>> text = "Hello world-\\nexample.\\nSecond example, an’ yeh’ll be fine."
        >>> prepare_english(text, rm_headers=False, rm_Hagrid_dialect=False)
        'Hello worldexample\\n.\\nSecond example, an’ yeh’ll be fine.'
        
        >>> prepare_english(12345, rm_headers=True, rm_Hagrid_dialect=True)
        Traceback (most recent call last):
        AttributeError: Invalid input
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        # remove hyphenations and unify hyphenated words
        # \1 - named groups
        # \w - unicode word characters
        text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
        text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)
        
        # remove unnecessary header texts, page numbers
        if(rm_headers) :
            text = re.sub('[0-9]+ HARRY  POTTER', '', text)
            text = re.sub('.*[A-Z] [0-9]+', '', text)
            text = re.sub('[—]', '', text)
        
        # replace Hagrid's dialect with the normal words
        # some words remain in text (e.g. askin’ - asking, etc.) but occur
        # too sporadic to handle this way
        if(rm_Hagrid_dialect) :
            text = text.replace(' an’ ', ' and ')
            text = text.replace(' don’ ', ' do not ')
            text = text.replace(' didn’ ', ' did not ')
            text = text.replace(' yeh ', ' you ')
            text = text.replace('Yeh ', 'You ')
            text = text.replace(' yeh.', ' you.')
            text = text.replace(" d’yeh ", 'Do you ')
            text = text.replace("D’yeh,", 'Do you ')
            text = text.replace(" D’yeh ", ' Do you ')
            text = text.replace(" D’yeh,", ' Do you ')
            text = text.replace(" yeh’d ", " you’d ")
            text = text.replace(" yeh’ve ", " you have ")
            text = text.replace(" yeh’ll ", " you will ")
            text = text.replace(' yer ', ' your ')
            text = text.replace(' ter ', ' to ')
            text = text.replace(' fer ', ' for ')
            text = text.replace('yeh', 'you')
            text = text.replace('Yeh', 'You')
         
        #  remove "er"-s from text which confuses topic models
        text = text.replace(' er ', '')
        
    return(text)


if __name__ == "__main__":
    import doctest
    doctest.testmod()