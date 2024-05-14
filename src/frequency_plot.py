# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 03:14:21 2023

@author: adamWolf
'''

from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


LANGUAGES = {'english', 'german', 'french', 'spanish'}

def word_frequency(text, text_language):
    """
    Analyzes the frequency of words in the input text and visualizes the top 100 most common words in a horizontal bar graph.

    Parameters:
        text (str): The input text to be analyzed for word frequency.
        text_language (str): The language of the input text. It must be one of the languages supported in the LANGUAGES set.

    Raises:
        AttributeError: If the input text is not a string or the text language is not supported.
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    elif text_language not in LANGUAGES:
        raise AttributeError('Language not supported')
    else:
        text = text.lower()
        cnt = Counter()
    
        # iterate over text splitted to words, and collect number of each word
        for text in text.split():
                cnt[text] += 1
    
        # Collect the most common 100 words
        most_common = cnt.most_common(100)
    
        word_freq = pd.DataFrame(most_common, columns=['words', 'count'])
        word_freq.head()
    
        fig, ax = plt.subplots(figsize=(12, 20))
    
        # Plot horizontal bar graph
        word_freq.sort_values(by = 'count').plot.barh(x = 'words',
                              y = 'count',
                              ax = ax,
                              color = 'brown')
        ax.set_title(f'Common Words Found in {text_language}')
        
        plt.show()