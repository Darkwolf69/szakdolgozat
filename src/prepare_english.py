# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 23:51:01 2023

@author: adamWolf

Prepare english text to additional normalizing
'''

import re


def prepare_english(text):
    
    # remove hyphenations and unify hyphenated words
    # \1 - named groups
    # \w - unicode word characters
    text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
    text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)
    
    #remove unnecessary header texts, page numbers and chapter words
    text = re.sub('[0-9]+ HARRY  POTTER','', text)
    text = re.sub('.*[A-Z] [0-9]+','', text)
    prepared_text = re.sub('[—]','',text)
    
    return(prepared_text)


#tried to convert 'stuttering' into just the word (e.g. S-s-sorry -> sorry)
#not working for some reason - writes the first match into every place
#(while by printing the iter values there are all of matches)
# =============================================================================
# iterate = re.findall(r'[^a-z][a-z]-[a-z]-[a-z]+', text, re.IGNORECASE)
# for iter in iterate:
#     print(iter)
#     text = re.sub(r'[^a-zA-Z][a-zA-Z]-[a-zA-Z]-[a-zA-Z]+', iter[5:], text)
# =============================================================================
