# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:55 2023

@author: adamWolf

Generate and display word lengths and word counts plots
from each chapters of texts of all languages
"""

import basic_functions
import matplotlib.pyplot as plt

num_of_chapters = 17
languages = ['english', 'german', 'french', 'spanish']

def chapter_stats(file_path):
    chapter_file = basic_functions.open_text(file_path)
    chapter_str = chapter_file.read()
    
    substr = 'chapters\\'
    file_name = file_path.partition(substr)[2]
    print(f'processing {file_name}')

    # split data into a list of words
    words = [x for x in chapter_str.split() if x]

    #filter remaining empty strings
    words = list(filter(None, words))

    # count the different lenghts using a dict
    dictionary = {}
    for x in words:
        word_length = len(x)
        dictionary[word_length] = dictionary.get(word_length, 0) + 1

    # retrieve the relevant info from the dict 
    lengths, counts = zip(*dictionary.items())

    # plot the relevant info
    plt.figure(figsize=(14,7))
    plt.bar(lengths, counts)
    plt.xticks(range(1, max(lengths)+1))
    plt.xlabel('Word lengths')
    plt.ylabel('Word counts')

    # what is the longest word
    plt.title(f'Longest word/words in {file_name}: ' + ' '.join(x for x in words if len(x)==max(lengths)))

    plt.show()
    
#calculating and plotting word lengths statistics of all chapters is all languages
for i in range(0, len(languages)):
    for j in range(1, num_of_chapters+1):
        if j < 10:
            chapter_stats(f'HP-{languages[i]}_chapters\\HP-{languages[i]}_chapter_0{j}.txt')
        else:
            chapter_stats(f'HP-{languages[i]}_chapters\\HP-{languages[i]}_chapter_{j}.txt')

                
