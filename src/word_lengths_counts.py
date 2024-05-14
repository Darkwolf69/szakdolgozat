# -*- coding: utf-8 -*-
'''
Created on Sat Nov 11 21:18:27 2023

@author: adamWolf
'''

import matplotlib.pyplot as plt


def word_lengths_counts(text, language):
    """
    Analyzes the distribution of word lengths in the provided text and generates a bar plot 
    showing the counts of words of each length. Also, displays the four longest words in the text.

    Args:
        text (str): The input text to be analyzed.
        language (str): The language of the text.

    Returns:
        matplotlib.pyplot: A bar plot showing the distribution of word lengths and the titles
        indicating the four longest words in the text.

    Raises:
        AttributeError: If the input text is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        # split data into a list of words
        words = [x for x in text.split() if x]
        
        # remove suffixes and prefixes (used just for english text)
        for suffix in ('’', '”'):
            words = [x.removesuffix(suffix) for x in words]
            
        for prefix in ('“', '‘'):
            words = [x.removeprefix(prefix) for x in words]
        
        #filter single qoutation marks
        if(language == 'english'):
            words = list(filter(lambda x: x != "'", words))
        
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
        plt.figure(figsize = (14,7))
        plt.bar(lengths, counts)
        plt.xticks(range(1, max(lengths) + 1))
        plt.xlabel('Word lengths')
        plt.ylabel('Word counts')
        
        # displaying the 3 longest words
        longests = sorted(words, key = len)[-4:]
        plt.title(f'The 4 longest words in {language} text:\n' + '\n'.join(reversed(longests) ), wrap = True)

        return plt
