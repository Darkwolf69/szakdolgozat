# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:47:53 2023

@author: adamWolf

Open clean texts, remove remaining unnecessary characters and remove newlines,
to prepare them for statistics analysis for each chapter
"""

import basic_functions
import re
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-english.txt')).read()

#remove hyphenations and unify hyphenated words
text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)

#remove unnecessary em-dash, header and footer texts, page numbers and chapter words
text = re.sub('[0-9]+ HARRY  POTTER','', text)
text = re.sub('.*[A-Z] [0-9]+','', text)
text = re.sub('[—]','',text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

# split data into a list of words
words = [x for x in text.split() if x]
# remove suffixes and prefixes
for suffix in ('’','”'):
    words = [x.removesuffix(suffix) for x in words]
for prefix in ('“','‘'):
    words = [x.removeprefix(prefix) for x in words]
clean_text = ' '.join(words)

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

basic_functions.write_text('HP-english_to_chapters.txt', clean_text)

#remove newline characters from texts (except for english)
text = (basic_functions.open_text('HP-german_clean.txt')).read()
text = re.sub('[\n]',' ',text)
basic_functions.write_text('HP-german_to_chapters.txt', text)

#unnecessary, since chapter texts and titles are in the same line, but done for integrity
text = (basic_functions.open_text('HP-french_clean.txt')).read()
text = re.sub('[\n]',' ',text)
basic_functions.write_text('HP-french_to_chapters.txt', text)

text = (basic_functions.open_text('HP-spanish_clean.txt')).read()
text = re.sub('[\n]',' ',text)
basic_functions.write_text('HP-spanish_to_chapters.txt', text)