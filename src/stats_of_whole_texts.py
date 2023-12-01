# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:46:10 2023

@author: adamWolf

Run all basic statistics analysis of whole texts
"""

import basic_functions
import remove_stopwords as rm_stopwords
import clear_HP_english
import clear_HP_french
import clear_HP_german
import clear_HP_spanish
import num_of_words_lines
import word_lengths_counts as len_cnt
import frequency_plot


clear_HP_english
clear_HP_german
clear_HP_french
clear_HP_spanish

num_of_words_lines

text_english = basic_functions.open_text('HP-english_clean.txt').read()
text_german = basic_functions.open_text('HP-german_clean.txt').read()
text_french = basic_functions.open_text('HP-french_clean.txt').read()
text_spanish = basic_functions.open_text('HP-spanish_clean.txt').read()

len_cnt.word_lengths_counts(text_english, 'english')
len_cnt.word_lengths_counts(text_german, 'german')
len_cnt.word_lengths_counts(text_french, 'french')
len_cnt.word_lengths_counts(text_spanish, 'spanish')

text_english = rm_stopwords.remove_stopwords(text_english, 'english')
frequency_plot.word_frequency(text_english, 'english')
text_german = rm_stopwords.remove_stopwords(text_german, 'german')
frequency_plot.word_frequency(text_german, 'german')
text_french = rm_stopwords.remove_stopwords(text_french, 'french')
frequency_plot.word_frequency(text_french, 'french')
text_spanish = rm_stopwords.remove_stopwords(text_spanish, 'spanish')
frequency_plot.word_frequency(text_spanish, 'spanish')







