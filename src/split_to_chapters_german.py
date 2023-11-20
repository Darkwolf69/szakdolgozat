# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 04:08:21 2023

@author: adamWolf
"""

import basic_functions
# import re

text = (basic_functions.open_text('HP-spanish_clean.txt')).read()

words = [x for x in text.split() if x]

clean_text = ' '.join(words)
basic_functions.write_text('HP-spanish_to_chapters.txt', clean_text)

# initializing split character
split_string = "CAPÍTULO"

# Prefix extraction before specific string
res = clean_text.rsplit(split_string, 16)[14]
# print(str(res))
# print(res)

res2 = res.split(' ', 2)[2]
print(res2)

basic_functions.write_text('HP-spanish_first_chapter.txt', res2)