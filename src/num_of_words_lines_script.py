# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 03:24:57 2023

@author: adamWolf

Generate plot for displaying number of words and lines
in whole texts of all languages
"""

import matplotlib.pyplot as plt

import basic_functions as base_fns
import prepare_english as prep_eng
import prepare_german as prep_ger
import prepare_french as prep_fre
import prepare_spanish as prep_esp
import remove_empty_lines as rm_empty_lines
import remove_characters as rm_chars
import expand_contractions as exp_contr


#read texts
english_text = base_fns.open_text('HP-english.txt').read()
german_text = base_fns.open_text('HP-german.txt').read()
french_text = base_fns.open_text('HP-french.txt').read()
spanish_text = base_fns.open_text('HP-spanish.txt').read()

#normalize texts
english_text = prep_eng.prepare_english(english_text)
english_text = rm_empty_lines.remove_empty_lines(english_text)
english_text = rm_chars.remove_unnecessary_chars(english_text)
english_text = exp_contr.expand_contractions(english_text)

german_text = prep_ger.prepare_german(german_text)
german_text = rm_empty_lines.remove_empty_lines(german_text)
german_text = rm_chars.remove_unnecessary_chars(german_text)

french_text = prep_fre.prepare_french(french_text)
french_text = rm_empty_lines.remove_empty_lines(french_text)
french_text = rm_chars.remove_unnecessary_chars(french_text)

spanish_text = prep_esp.prepare_spanish(spanish_text)
spanish_text = rm_empty_lines.remove_empty_lines(spanish_text)
spanish_text = rm_chars.remove_unnecessary_chars(spanish_text)

#count and store line and word numbers
numw_eng = base_fns.number_of_words_in_text(english_text)
numw_ger = base_fns.number_of_words_in_text(german_text)
numw_fr = base_fns.number_of_words_in_text(french_text)
numw_sp = base_fns.number_of_words_in_text(spanish_text)

numl_eng = base_fns.line_number_of_text(english_text)
numl_ger = base_fns.line_number_of_text(german_text)
numl_fr = base_fns.line_number_of_text(french_text)
numl_sp = base_fns.line_number_of_text(spanish_text)

# preparing data on which bar chart will be plot
languages = ('English', 'German', 'French', 'Spanish')
word_counts = (numw_eng, numw_ger, numw_fr, numw_sp)
line_counts = (numl_eng, numl_ger, numl_fr, numl_sp)


# create and show two subplots with the received data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 7), sharey = True, layout = 'constrained')

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