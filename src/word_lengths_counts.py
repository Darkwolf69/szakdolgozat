# -*- coding: utf-8 -*-
'''
Created on Sat Nov 11 21:18:27 2023

@author: adamWolf

Open the prepared clean english text file,
calculate and create a plot displaying word lengths, word counts and longest words
'''

import matplotlib.pyplot as plt


def word_lengths_counts(text, language):
    
    # split data into a list of words
    words = [x for x in text.split() if x]
    
    # remove suffixes and prefixes (used just for english text)
    for suffix in ('’','”'):
        words = [x.removesuffix(suffix) for x in words]
        
    for prefix in ('“','‘'):
        words = [x.removeprefix(prefix) for x in words]
    
    #filter remaining empty strings
    words = list(filter(None, words))
    
    # count the different lengths using a dict
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
    
    # displaying the longest word
    plt.title(f'Longest word in {language} text: ' + ' '.join(x for x in words if len(x)==max(lengths)))
    
    plt.show()
