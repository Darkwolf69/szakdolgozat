# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:10:30 2023

@author: adamWolf
"""

import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text
    for word in text])
    return text