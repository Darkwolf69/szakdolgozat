# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:22:42 2023

@author: adamWolf
"""

import os
import basic_functions as base_fns
import prepare_english as prep_eng
import remove_empty_lines as rm_empty_lns
import make_topics as mk_tps


text = (base_fns.open_text('HP-english_original.txt')).read()
text = prep_eng.prepare_english(text, rm_headers=True, rm_Hagrid_dialect=True)
text = rm_empty_lns.remove_empty_lines(text)

num_of_chapters = 17

#generate folders for storing chapter files
if not os.path.exists('C:\\Users\\farka\\Harry_Potter\\chapters_for_topics\\'):
    os.makedirs('C:\\Users\\farka\\Harry_Potter\\chapters_for_topics\\')

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
file_path = 'C:\\Users\\farka\\Harry_Potter\\chapters_for_topics\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\chapters_for_topics\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_1_chapters.append(data)
        
HP_2_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\chamber_secrets_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\chamber_secrets_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_2_chapters.append(data)
        
HP_3_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\prisoner_azkaban_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\prisoner_azkaban_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_3_chapters.append(data)
        
HP_4_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\goblet_fire_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\goblet_fire_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_4_chapters.append(data)
        
HP_5_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\order_phoenix_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\order_phoenix_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_5_chapters.append(data)
        
HP_6_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\half_blood_prince_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\half_blood_prince_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_6_chapters.append(data)
        
HP_7_chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\deathly_hallows_chapters\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\deathly_hallows_chapters\\{file_name}'
    with open(file_path, encoding='utf-8-sig', errors='ignore', mode='r+') as f:
        data = f.read()
        HP_7_chapters.append(data)


print('Topics of the Philosophers Stone:')
mk_tps.make_topics(HP_1_chapters, 'english', 5)
print('\nTopics of the Chamber of Secrets:')
mk_tps.make_topics(HP_2_chapters, 'english', 5)
print('\nTopics of the Prisoner Of Azkaban:')
mk_tps.make_topics(HP_3_chapters, 'english', 5)
print('\nTopics of the Goblet of Fire:')
mk_tps.make_topics(HP_4_chapters, 'english', 5)
print('\nTopics of the Order of The Phoenix:')
mk_tps.make_topics(HP_5_chapters, 'english', 5)
print('\nTopics of the Half-Blood Prince:')
mk_tps.make_topics(HP_6_chapters, 'english', 5)
print('\nTopics of the Deathly Hallows:')
mk_tps.make_topics(HP_7_chapters, 'english', 5)
