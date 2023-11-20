# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 03:14:21 2023

@author: adamWolf

Display word frequency plot of given text
'''

import basic_functions
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def word_frequency(textFile, text_language):
    data = (basic_functions.open_text(textFile)).read()
    data = data.lower()
    
    cnt = Counter()
    
    for data in data.split():
        cnt[data] += 1
        
    # See most common 100 words
    most_common = cnt.most_common(100)
    
    word_freq = pd.DataFrame(cnt.most_common(100), columns=['words', 'count'])
    word_freq.head()
    
    fig, ax = plt.subplots(figsize=(12, 20))
    
    # Plot horizontal bar graph
    word_freq.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color='brown')
    ax.set_title(f'Common Words Found in {text_language}')
    
    plt.show()