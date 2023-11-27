# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:52:39 2023

@author: adamWolf

gets a text corpus and returns a corpus normalized with the selected options
possible methods: contraction_expansion, text_lower_case, text_lemmatization,
special_char_removal, stopword_removal, remove_digits
"""

import basic_functions
import re
import expand_contractions
import lemmatize_text
import remove_special_characters
import remove_stopwords


def normalize_corpus(
    corpus,
    contraction_expansion,
    text_lower_case,
    text_lemmatization,
    special_char_removal,
    stopword_removal,
    remove_digits
):
    contraction_expansion = contraction_expansion
    text_lower_case = text_lower_case
    text_lemmatization = text_lemmatization
    special_char_removal = special_char_removal
    stopword_removal = stopword_removal
    remove_digits = remove_digits
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions.expand_contractions(doc)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text.lemmatize_text(doc)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters.remove_special_characters(doc, remove_digits=remove_digits)
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords.remove_stopwords(doc, is_lower_case=text_lower_case)
        normalized_corpus.append(doc)
                
    return normalized_corpus


text = (basic_functions.open_text('HP-english.txt')).read()
print(len(text))


normalized_text = normalize_corpus([text], True, False, False, True, False, False)[0]
print(normalized_text)




