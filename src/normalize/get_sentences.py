# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:52:51 2023

@author: adamWolf
"""

import nltk


def get_sentences(text):
    """
    Tokenizes the input text into sentences using sentence tokenizer of NLTK module.

    Parameters:
        text (str): The input text to be tokenized into sentences.

    Returns:
        list: A list of sentences extracted from the input text.

    Raises:
        AttributeError: If the input text is not a string.
        
    
    Examples:
        >>> get_sentences("Hello world. This is a test.")
        ['Hello world.', 'This is a test.']
        
        >>> get_sentences("Here is a sentence! And another one? Yes, indeed.")
        ['Here is a sentence!', 'And another one?', 'Yes, indeed.']
        
        >>> get_sentences("Single sentence without period")
        ['Single sentence without period']
        
        >>> get_sentences("")
        []

        >>> get_sentences(12345)
        Traceback (most recent call last):
        AttributeError: Invalid input
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        sentences = nltk.sent_tokenize(text)
        
    return sentences


if __name__ == "__main__":
    import doctest
    doctest.testmod()