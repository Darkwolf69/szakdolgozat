# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 02:09:16 2023

@author: adamWolf

Display word frequency plot of texts in all languages
'''

import frequency_plot

frequency_plot.word_frequency('HP-english_clean.txt', 'english')
frequency_plot.word_frequency('HP-german_clean.txt', 'german')
frequency_plot.word_frequency('HP-french_clean.txt', 'french')
frequency_plot.word_frequency('HP-spanish_clean.txt', 'spanish')