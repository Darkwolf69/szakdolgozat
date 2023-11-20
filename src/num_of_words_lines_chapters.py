# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:23:18 2023

@author: adamWolf

Generate plot (?) for displaying number of words and lines in chapters of all languages
"""

import os

num_of_chapters = 17
languages = ['english', 'german', 'french', 'spanish']

#TODO: figure out how to collect and display stats of all chapters
for i in range(0, num_of_chapters):
    if i < 9:
        subdir = 'HP-english_chapters'
        filename = f'HP-english_chapter_0{i+1}.txt'
        path = os.path.join(subdir, filename)
        with open(path, 'r', encoding='utf-8-sig') as f: 
            text = f.read()
    else:
        subdir = 'HP-english_chapters'
        filename = f'HP-english_chapter_{i+1}.txt'
        path = os.path.join(subdir, filename)
        with open(path, 'w', encoding='utf-8-sig') as f: 
            text = f.read()
            

# =============================================================================
# tuples = (2, 4, 6, 8)
# list1 = list(tuples)
# list1.append(12)
# result = tuple(list1)
# print(result)
# =============================================================================
