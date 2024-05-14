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
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        sentences = nltk.sent_tokenize(text)
        
    return sentences
