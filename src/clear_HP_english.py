# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 23:51:01 2023

@author: adamWolf

Open and clear english text file, to prepare for statistic analysis.
Output is a text containing just words including chapter titles,
newline characters and quotation marks.
'''

import basic_functions
import re
import remove_characters as rm_chars
import remove_empty_lines
import expand_contractions as exp_cons


text = (basic_functions.open_text('HP-english.txt')).read()

# remove hyphenations and unify hyphenated words
# \1 - named groups
# \w - unicode word characters
text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)

#remove unnecessary header texts, page numbers and chapter words
text = re.sub('[0-9]+ HARRY  POTTER','', text)
text = re.sub('.*[A-Z] [0-9]+','', text)
text = re.sub('[—]','',text)
text = re.sub('— CHAPTER [A-Z]+ —','', text)


#try to convert 'stuttering' into just the word (e.g. S-s-sorry -> sorry)
#not working for some reason - writes the first match into every place
#(while by printing the iter values there are all of matches)
# =============================================================================
# iterate = re.findall(r'[^a-z][a-z]-[a-z]-[a-z]+', text, re.IGNORECASE)
# for iter in iterate:
#     print(iter)
#     text = re.sub(r'[^a-zA-Z][a-zA-Z]-[a-zA-Z]-[a-zA-Z]+', iter[5:], text)
# =============================================================================
    

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)
#remove unnecessary characters (dots, etc.)
text = rm_chars.remove_unnecessary_chars(text)

#expand contractions
text = exp_cons.expand_contractions(text)

basic_functions.write_text('HP-english_clean.txt', text)