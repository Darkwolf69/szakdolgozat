# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:51:38 2023

@author: adamWolf
"""

import nltk
import re

def normalize_document(doc):
    
    stop_words = nltk.corpus.stopwords.words('english')
    # lower case and remove special characters/whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', " ", doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc