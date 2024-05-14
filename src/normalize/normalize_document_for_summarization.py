# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:51:38 2023

@author: adamWolf
"""

import re

import nltk


def normalize_document(text):
    """
    Normalize the given text by converting it to lowercase, removing special characters, and filtering out stopwords.

    Parameters:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.

    Raises:
        AttributeError: If the input is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        stop_words = nltk.corpus.stopwords.words('english')
        # lower case and remove special characters/whitespaces
        # igorecase and ASCII only matching
        text = re.sub(r'[^a-zA-Z\s]',  " ", text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        # tokenize textument
        tokens = nltk.word_tokenize(text)
        # filter stopwords out of textument
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create textument from filtered tokens
        text = ' '.join(filtered_tokens)
        
        return text