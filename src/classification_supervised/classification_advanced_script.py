# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:47:33 2024

@author: adamWolf

Call modules of advanced text classification
"""

import numpy as np

from classification_prepare import make_chapters_to_classification
from classification_advanced_prepare import prep_classification_advanced
from classification_advanced import classification_advanced


make_chapters_to_classification()

# reading prepared train and test corpus
with open('datasets_list.npy', 'rb') as f:
    train_corpus = np.load(f, allow_pickle=True)
    test_corpus = np.load(f, allow_pickle=True)
    train_label_nums = np.load(f, allow_pickle=True)
    test_label_nums = np.load(f, allow_pickle=True)
    train_label_names = np.load(f, allow_pickle=True)
    test_label_names = np.load(f, allow_pickle=True)
    
with open('datasets_list_Word2Vec.npy', 'rb') as f:
    avg_w2v_train_features = np.load(f, allow_pickle=True)
    avg_w2v_test_features = np.load(f, allow_pickle=True)
    
with open('datasets_list_GloVe.npy', 'rb') as f:
    train_glove_features = np.load(f, allow_pickle=True)
    test_glove_features = np.load(f, allow_pickle=True)
    
with open('datasets_list_FastText.npy', 'rb') as f:
    avg_ft_train_features = np.load(f, allow_pickle=True)
    avg_ft_test_features = np.load(f, allow_pickle=True)
    

prep_classification_advanced()
classification_advanced(train_corpus, test_corpus, train_label_nums,
                        test_label_nums, train_label_names, test_label_names,
                        avg_w2v_train_features, avg_w2v_test_features,
                        train_glove_features, test_glove_features,
                        avg_ft_train_features, avg_ft_test_features)
