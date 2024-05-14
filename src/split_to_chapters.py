# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:55 2023

@author: adamWolf
"""

import os
import re

import basic_functions
import prepare_english as prep_eng


def make_chapters(file_name, split_string, language, num_of_chapters):
    """
    Generate chapter files from a given text file.

    This function reads a text file, splits it into chapters based on the provided split string,
    prepares the chapters based on the specified language, and generates individual chapter files.

    Paramaters:
        file_name (str): The name of the text file.
        split_string (str): The string to split chapters on.
        language (str): The language of the text file.
        num_of_chapters (int): The number of chapters in the text file.

    Returns:
        None

    Raises:
        IOError: If there is an issue reading from the file.
    """
    
    LANGUAGES = ['english', 'german', 'french', 'spanish']
    
    try:
        text = (basic_functions.open_text(file_name)).read()
    except IOError:
        print("Something went wrong when reading from the file")
    else:
        if (language == 'english'):
            text = prep_eng.prepare_english(text, True, True)
        
        #generate folders for storing chapter files in all languages
        for i in range(0, len(LANGUAGES)):
            if not os.path.exists(f'my_path_to_save_chapters\\{file_name[:-4]}_chapters\\'):
                os.makedirs(f'my_path_to_save_chapters\\{file_name[:-4]}_chapters\\')
    
        # Prefix extraction before specific string (removes 'chapter xy' words from chapters)
        # This regular expression applicable only for format e.g. "Chapter 1"
        res = re.split(rf"{split_string} [0-9]+", text)
    
        for i in range(0, num_of_chapters):
            subdir = f'{file_name[:-4]}_chapters'
            if i < 9:
                filename = f'{file_name[:-4]}_chapter_0{i+1}.txt'
            else:
                filename = f'{file_name[:-4]}_chapter_{i+1}.txt'
                
            path = os.path.join(subdir, filename)
            with open(path, 'w', encoding = 'utf-8-sig') as f: 
                f.write(res[i + 1])
                
            
make_chapters('HP-english.txt', 'CHAPTER', 'english', 17)
make_chapters('HP-german.txt', 'KAPITEL', 'german', 17)
make_chapters('HP-french.txt', 'CHAPITRE', 'french', 17)
make_chapters('HP-spanish.txt', 'CAPÍTULO', 'spanish', 17)

make_chapters('philosophers_stone.txt', 'Chapter', 'english', 17)
make_chapters('chamber_secrets.txt', 'Chapter', 'english', 18)
make_chapters('prisoner_azkaban.txt', 'Chapter', 'english', 22)
make_chapters('goblet_fire.txt', 'Chapter', 'english', 37)
make_chapters('order_phoenix.txt', 'Chapter', 'english', 38)
make_chapters('half_blood_prince.txt', 'Chapter', 'english', 30)
make_chapters('deathly_hallows.txt', 'Chapter', 'english', 36)
