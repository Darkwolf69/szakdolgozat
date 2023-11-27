# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:42:21 2023

@author: adamWolf
"""

import basic_functions
import nltk
import normalize_document
import numpy as np
from gensim.models import Word2Vec

text = (basic_functions.open_text('HP-english_original.txt')).read()

normalize_corpus = np.vectorize(normalize_document)

# get sentences in the document
sentences = nltk.sent_tokenize(text)
newSentences = []

#remove newline characters
for sub in sentences:
    newSentences.append(sub.replace("\n", ""))
    

#initialize the model with the input parameters
model = Word2Vec(newSentences, vector_size=10, min_count=1)
print(model)

#the sentences are converted into vectors of size 10
sent1 = np.mean(np.array(model.wv[model[0].split(' ')]), axis=0)
sent2 = np.mean(np.array(model.wv[model[1].split(' ')]), axis=0)
sent3 = np.mean(np.array(model.wv[model[2].split(' ')]), axis=0)

print(sent1)
    
