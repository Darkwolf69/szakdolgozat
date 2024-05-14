# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:52:39 2023

@author: adamWolf
"""

import re

import expand_contractions
import lemmatize_text
import remove_special_characters
import remove_stopwords as rm_stopwds


def normalize_corpus(
    corpus,
    language,
    contraction_expansion,
    text_lower_case,
    text_lemmatization,
    special_char_removal,
    stopword_removal,
    remove_digits,
):
    """
    Normalize a corpus of text documents based on specified normalization techniques.

    Parameters:
        corpus (list of str): The list of text documents to be normalized.
        language (str): The language of the text documents.
        contraction_expansion (bool): Whether to expand contractions in the text.
        text_lower_case (bool): Whether to convert the text to lowercase.
        text_lemmatization (bool): Whether to lemmatize the text.
        special_char_removal (bool): Whether to remove special characters.
        stopword_removal (bool): Whether to remove stopwords.
        remove_digits (bool): Whether to remove digits.

    Returns:
        list of str: The normalized corpus.

    Raises:
        None.

    """
    contraction_expansion = contraction_expansion
    text_lower_case = text_lower_case
    text_lemmatization = text_lemmatization
    special_char_removal = special_char_removal
    stopword_removal = stopword_removal
    remove_digits = remove_digits
    normalized_corpus = []
    
    # normalize each text in the corpus
    for text in corpus:
        # expand contractions
        if contraction_expansion:
            text = expand_contractions.expand_contractions(text)
        # lowercase the text
        if text_lower_case:
            text = text.lower()
            # remove extra newlines
            text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        # lemmatize text
        if text_lemmatization:
            text = lemmatize_text.lemmatize_text(text)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            text = special_char_pattern.sub(" \\1 ", text)
            text = remove_special_characters.remove_special_characters(text, remove_digits = remove_digits)
            # remove extra whitespace
            text = re.sub(' +', ' ', text)
        # remove stopwords
        if stopword_removal:
            text = rm_stopwds.remove_stopwords(text, language, is_lower_case = text_lower_case)
        normalized_corpus.append(text)
                
    return normalized_corpus





