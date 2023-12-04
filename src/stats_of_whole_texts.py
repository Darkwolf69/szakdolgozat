# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:46:10 2023

@author: adamWolf

Run all basic statistics analysis of whole texts
"""

import basic_functions as base_fns

import prepare_english as prep_eng
import prepare_german as prep_ger
import prepare_french as prep_fre
import prepare_spanish as prep_esp

import remove_empty_lines as rm_empty_lines
import remove_characters as rm_chars
import remove_special_characters as rm_all_chars
import expand_contractions as exp_contr
import remove_stopwords as rm_stopwords

import num_of_words_lines
import word_lengths_counts as len_cnt
import frequency_plot


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
english_text = english_text.replace("'", '')

german_text = prep_ger.prepare_german(german_text)
german_text = rm_empty_lines.remove_empty_lines(german_text)
german_text = rm_chars.remove_unnecessary_chars(german_text)
german_text = german_text.replace('’', '')

french_text = prep_fre.prepare_french(french_text)
french_text = rm_empty_lines.remove_empty_lines(french_text)
french_text = rm_chars.remove_unnecessary_chars(french_text)
french_text = french_text.replace("'", '')

spanish_text = prep_esp.prepare_spanish(spanish_text)
spanish_text = rm_empty_lines.remove_empty_lines(spanish_text)
spanish_text = rm_chars.remove_unnecessary_chars(spanish_text)


num_of_words_lines

len_cnt.word_lengths_counts(english_text, 'english')
len_cnt.word_lengths_counts(german_text, 'german')
len_cnt.word_lengths_counts(french_text, 'french')
len_cnt.word_lengths_counts(spanish_text, 'spanish')


text_english = rm_stopwords.remove_stopwords(english_text, 'english')
frequency_plot.word_frequency(text_english, 'english')
text_german = rm_stopwords.remove_stopwords(german_text, 'german')
frequency_plot.word_frequency(text_german, 'german')
text_french = rm_stopwords.remove_stopwords(french_text, 'french')
frequency_plot.word_frequency(text_french, 'french')
text_spanish = rm_stopwords.remove_stopwords(spanish_text, 'spanish')
frequency_plot.word_frequency(text_spanish, 'spanish')









