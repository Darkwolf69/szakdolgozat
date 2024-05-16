# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:10:30 2023

@author: adamWolf
"""

import spacy


nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    """
    Lemmatizes the words in the given text using the English language model of SpaCy module.

    Paramaters:
        text (str): The input text to be lemmatized.

    Returns:
        str: The lemmatized text.

    Raises:
        AttributeError: If the input is not a string.
        
    
    Examples:
        >>> lemmatize_text("The striped bats are hanging on their feet for best")
        'the stripe bat be hang on their foot for good'

        >>> lemmatize_text("She enjoys playing with her cats")
        'she enjoy play with her cat'

        >>> lemmatize_text(12345)
        Traceback (most recent call last):
        AttributeError: Invalid input

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text
        for word in text])
        return text


if __name__ == "__main__":
    import doctest
    doctest.testmod()