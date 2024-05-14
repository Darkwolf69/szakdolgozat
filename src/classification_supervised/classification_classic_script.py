# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:56:09 2024

@author: adamWolf

Call modules of classic text classification
"""

import json

import numpy as np

from classification_prepare import make_chapters_to_classification
from classification_classic import classification_classic
from classification_classic_show_results import show_classification_results


make_chapters_to_classification()

# reading prepared train and test corpus
with open('datasets_list.npy', 'rb') as f:
    train_corpus = np.load(f, allow_pickle=True)
    test_corpus = np.load(f, allow_pickle=True)
    train_label_nums = np.load(f, allow_pickle=True)
    test_label_nums = np.load(f, allow_pickle=True)
    train_label_names = np.load(f, allow_pickle=True)
    test_label_names = np.load(f, allow_pickle=True)
    

classification_classic(train_corpus, test_corpus, train_label_nums,
                           test_label_nums, train_label_names, test_label_names)

with open('classification_classic_results.txt', 'r') as file:
    results = json.load(file)

if(results):
    show_classification_results()
else:
    print('Something went wrong, nothing to show')