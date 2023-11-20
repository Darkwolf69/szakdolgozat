# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:55 2023

@author: adamWolf

Generate folders and writing chapters is separate files into appropriate folders
"""

import basic_functions
import os

def make_chapters(file_name, split_string, language):
    
    num_of_chapters = 17
    languages = ['english', 'german', 'french', 'spanish']
    
    text = (basic_functions.open_text(file_name)).read()
    
    # initializing split string
    split_str = split_string
    
    #generate folders for storing chapter files in all languages
    for i in range(0, len(languages)):
        if not os.path.exists(f'C:\\Users\\farka\\Harry_Potter\\HP-{languages[i]}_chapters\\'):
            os.makedirs(f'C:\\Users\\farka\\Harry_Potter\\HP-{languages[i]}_chapters\\')

    # Prefix extraction before specific string (removes 'chapter xy' words from chapters)
    for i in range(0, num_of_chapters):
        res = text.rsplit(split_str, 16)[i]
        res2 = res.split(' ', 2)[2]
        if i < 9:
            subdir = f'HP-{language}_chapters'
            filename = f'HP-{language}_chapter_0{i+1}.txt'
            path = os.path.join(subdir, filename)
            with open(path, 'w', encoding='utf-8-sig') as f: 
                f.write(res2)
        else:
            subdir = f'HP-{language}_chapters'
            filename = f'HP-{language}_chapter_{i+1}.txt'
            path = os.path.join(subdir, filename)
            with open(path, 'w', encoding='utf-8-sig') as f: 
                f.write(res2)

make_chapters('HP-english_to_chapters.txt', 'CHAPTER', 'english')
make_chapters('HP-german_to_chapters.txt', 'KAPITEL', 'german')
make_chapters('HP-french_to_chapters.txt', 'CHAPITRE', 'french')
make_chapters('HP-spanish_to_chapters.txt', 'CAPÍTULO', 'spanish')