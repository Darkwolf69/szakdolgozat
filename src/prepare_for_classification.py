# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 01:23:23 2024

@author: adamWolf

Read all 7 Harry Potter books from files,
classified its chapters manually to 4 predefined topics,
split the books to chapters,
make a corpus containing all of chapters,
normalize the corpus,
returns with the test and train datasets for the classifier system

"""

import basic_functions
import os
import prepare_english as prep_eng
import re
import numpy as np
import pandas as pd
from normalize_texts import normalize_corpus
from sklearn.model_selection import train_test_split
from collections import Counter

def make_chapters_to_classification():
    
    # initializing necessary data models
    split_string = 'Chapter'
    
    file_names = [
        ['philosophers_stone.txt', 17],
        ['chamber_secrets.txt', 18],
        ['prisoner_azkaban.txt', 22],
        ['goblet_fire.txt', 37],
        ['order_phoenix.txt', 38],
        ['half_blood_prince.txt', 30],
        ['deathly_hallows.txt', 37]
    ]
    
    all_chapters_list = []
    
    labels_map = {
        0: 'muggle_world',
        1: 'magic_outside_Hogwarts',
        2: 'Voldemort_story_arch',
        3: 'Hogwarts_classroom_quidditch',
    }
    
    target_list = [0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 3, 2,
                   0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 2, 2, 3,
                   0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1, 3, 0,
                   2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 1, 1, 2, 2, 3, 3, 2, 2, 3, 2, 1,
                   0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 2, 1, 1, 3, 3, 3, 1, 3, 3, 1, 3, 2, 3, 1, 1, 2, 2, 2,
                   1, 2, 0, 1, 1, 2, 1, 3, 3, 2, 3, 3, 2, 3, 2, 1, 2, 3, 3, 2, 3, 3, 2, 3, 2, 2, 3, 2, 3, 1,
                   2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 3, 3, 3, 2, 1, 2, 1, 2, 1
    ]
    
    target = np.array(target_list)
    
    target_names = [
        'muggle_world',
        'magic_outside_Hogwarts',
        'Voldemort_story_arch',
        'Hogwarts_classroom_quidditch'
    ]
    
    
    # split books to chapters
    for i in range(len(file_names)) :
        text = (basic_functions.open_text(file_names[i][0])).read()
        text = prep_eng.prepare_english(text, False, True)
        
        # Prefix extraction before specific string (removes 'chapter xy' words from chapters)
        # This regular expression is applicable only for format e.g. "Chapter 1"
        res = re.split(rf"{split_string} [0-9]+", text)
        # print(res[0])
        
        for i in range(0, file_names[i][1]):
            all_chapters_list.append(res[i+1])
    
    
    # build the dataframe
    corpus, target_labels, target_names = (all_chapters_list, target,
                                           [labels_map[label] for label in target])
    
    data_df = pd.DataFrame({'Article': corpus, 'Target Label': target_labels,
                            'Target Name': target_names})
    # print(data_df)
    # data_df.head(10)
    
    # normalize the corpus
    normalized_corpus = normalize_corpus(corpus=data_df['Article'], language='english',
                                         contraction_expansion=True, text_lower_case=True,
                                         text_lemmatization=True, special_char_removal=True,
                                         stopword_removal=True, remove_digits=False)
    
    print('normalized corpus: ', normalized_corpus)
    data_df['Clean Article'] = normalized_corpus
    data_df = data_df[['Article', 'Clean Article', 'Target Label', 'Target Name']]
    # print(data_df)
    
    
    # build train and test datasets
    train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(np.array(data_df['Clean Article']),
    np.array(data_df['Target Label']),
    np.array(data_df['Target Name']),
    test_size=0.33, random_state=42)
    print('Train and test corpus shapes: ', train_corpus.shape, test_corpus.shape)
    print(type(train_corpus), type(test_corpus), type(train_label_nums),
          type(test_label_nums), type(train_label_names), type(test_label_names))
    
    # distribution of the chapters by topics
    #  TODO - add its results to the thesis work too
    trd = dict(Counter(train_label_names))
    tsd = dict(Counter(test_label_names))
    distr_df = (pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
    columns=['Target Label', 'Train Count', 'Test Count']).sort_values
     (by=['Train Count', 'Test Count'],ascending=False))
    print('distr_df:\n', distr_df)
    
    
    with open('datasets_list.npy', 'wb') as f:
        np.save(f, train_corpus)
        np.save(f, test_corpus)
        np.save(f, train_label_nums)
        np.save(f, test_label_nums)
        np.save(f, train_label_names)
        np.save(f, test_label_names)
        
        
make_chapters_to_classification()