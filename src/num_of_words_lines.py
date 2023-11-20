# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 03:24:57 2023

@author: adamWolf

Generate plot for displaying number of words and lines in whole texts of all languages
"""

import basic_functions as base_fns
import matplotlib.pyplot as plt

#returns total word numbers of the given text file
def num_words(file_name, file_language):
    return base_fns.number_of_words_in_text(base_fns.open_text(file_name))

#returns total line numbers of the given text file
def num_lines(file_name, file_language):
    return base_fns.line_number_of_text(base_fns.open_text(file_name))

languages = ['english','german','french','spanish']

#count and store line and word numbers
numw_eng = num_words('HP-english_clean.txt', 'english')
numw_ger = num_words('HP-german_clean.txt', 'german')
numw_fr = num_words('HP-french_clean.txt', 'french')
numw_sp = num_words('HP-spanish_clean.txt', 'spanish')

numl_eng = num_lines('HP-english_clean.txt', 'english')
numl_ger = num_lines('HP-german_clean.txt', 'german')
numl_fr = num_lines('HP-french_clean.txt', 'french')
numl_sp = num_lines('HP-spanish_clean.txt', 'spanish')

# preparing data on which bar chart will be plot
languages = ('English', 'German', 'French', 'Spanish')
word_counts = (numw_eng, numw_ger, numw_fr, numw_sp)
line_counts = (numl_eng, numl_ger, numl_fr, numl_sp)

# create and show two subplots with the received data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7), sharey=True, layout='constrained')

ax1.bar(languages, word_counts)
ax2.bar(languages, line_counts)

for i in range(len(languages)):
    ax1.text(i, word_counts[i], word_counts[i], ha = 'center', verticalalignment = 'bottom')

for i in range(len(languages)):
    ax2.text(i, line_counts[i], line_counts[i], ha = 'center', verticalalignment = 'bottom')

ax1.set_title('Number of words')
ax2.set_title('Number of lines')
ax1.set_ylabel('Word counts')

fig.supxlabel('Languages')
fig.suptitle('Basic statistics of the Harry Potter book in 4 languages')

plt.show()