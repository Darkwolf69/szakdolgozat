# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:14:08 2023

@author: adamWolf
"""

import nltk
from nltk.tokenize.toktok import ToktokTokenizer


tokenizer = ToktokTokenizer()

def remove_stopwords(text, language, is_lower_case=False):
    """
    Remove stopwords from a text string.

    Parameters:
        text (str): The input text string from which stopwords will be removed.
        language (str): The language of the text for selecting the appropriate stopwords list.
        is_lower_case (bool): Whether the text is already lowercased (default is False).

    Returns:
        str: The text string with stopwords removed.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        stopword_list = nltk.corpus.stopwords.words(language)
        
        # to keep negation if there is any in bi-grams
        if (language == 'english'):
            stopword_list.remove('no')
            stopword_list.remove('not')
    
            
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        
        return filtered_text
