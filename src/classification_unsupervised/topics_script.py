# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:22:42 2023

@author: adamWolf

Process text files, prepare them for topic modeling, generate chapters, and extract topics.
This function reads text files of each chapter of "Harry Potter" books, prepares them for topic modeling,
generates chapters, and extracts topics using Latent Semantic Indexing with Singular Value Decomposition (LSI_SVD).

Returns:
    None
"""

import os

import basic_functions as base_fns
import prepare_english as prep_eng
import remove_empty_lines as rm_empty_lns
import topics_make as mk_tps


try:
    text = (base_fns.open_text('HP-english_original.txt')).read()
except IOError:
    print("Something went wrong when reading from the file")
else:
    text = prep_eng.prepare_english(text, rm_headers = True, rm_Hagrid_dialect = True)
    text = rm_empty_lns.remove_empty_lines(text)
    
    num_of_chapters = 17
    
    #generate folders for storing chapter files
    if not os.path.exists('my_path_to_save_topics\\chapters_for_topics\\'):
        os.makedirs('my_path_to_save_topics\\chapters_for_topics\\')
    
    # Prefix extraction before specific string (removes 'chapter xy' words from chapters)
    # generate chapter files for Philisophers Stone
    for i in range(0, num_of_chapters):
        res = text.rsplit('CHAPTER', 16)[i]
        res2 = res.split(' ', 2)[2]
        if i < 9:
            subdir = 'chapters_for_topics'
            filename = f'HP_chapter_0{i+1}.txt'
            path = os.path.join(subdir, filename)
            with open(path, 'w', encoding='utf-8-sig') as f: 
                f.write(res2)
        else:
            subdir = 'chapters_for_topics'
            filename = f'HP_chapter_{i+1}.txt'
            path = os.path.join(subdir, filename)
            with open(path, 'w', encoding='utf-8-sig') as f: 
                f.write(res2)
    
    #load the created chapters into a list
    HP_1_chapters = []
    file_path = 'my_path_to_save_topics\\chapters_for_topics\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\chapters_for_topics\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_1_chapters.append(data)
            
    HP_2_chapters = []
    file_path = 'my_path_to_save_topics\\chamber_secrets_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\chamber_secrets_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_2_chapters.append(data)
            
    HP_3_chapters = []
    file_path = 'my_path_to_save_topics\\prisoner_azkaban_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\prisoner_azkaban_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_3_chapters.append(data)
            
    HP_4_chapters = []
    file_path = 'my_path_to_save_topics\\goblet_fire_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\goblet_fire_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_4_chapters.append(data)
            
    HP_5_chapters = []
    file_path = 'my_path_to_save_topics\\order_phoenix_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\order_phoenix_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_5_chapters.append(data)
            
    HP_6_chapters = []
    file_path = 'my_path_to_save_topics\\half_blood_prince_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\half_blood_prince_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_6_chapters.append(data)
            
    HP_7_chapters = []
    file_path = 'my_path_to_save_topics\\deathly_hallows_chapters\\'
    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        file_path = f'my_path_to_save_topics\\deathly_hallows_chapters\\{file_name}'
        with open(file_path, encoding = 'utf-8-sig', errors = 'ignore', mode = 'r+') as f:
            data = f.read()
            HP_7_chapters.append(data)
    
    
    print('Topics of the Philosophers Stone:')
    mk_tps.topics_make(HP_1_chapters, 'english', 5)
    print('\nTopics of the Chamber of Secrets:')
    mk_tps.topics_make(HP_2_chapters, 'english', 5)
    print('\nTopics of the Prisoner Of Azkaban:')
    mk_tps.topics_make(HP_3_chapters, 'english', 5)
    print('\nTopics of the Goblet of Fire:')
    mk_tps.topics_make(HP_4_chapters, 'english', 5)
    print('\nTopics of the Order of The Phoenix:')
    mk_tps.topics_make(HP_5_chapters, 'english', 5)
    print('\nTopics of the Half-Blood Prince:')
    mk_tps.topics_make(HP_6_chapters, 'english', 5)
    print('\nTopics of the Deathly Hallows:')
    mk_tps.topics_make(HP_7_chapters, 'english', 5)
