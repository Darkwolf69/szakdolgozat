# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:02:34 2023

@author: adamWolf
'''

import basic_functions
import re
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-german.txt')).read()

#remove unnecessary footer texts and page numbers
text = re.sub('HARRY POTTER [a-z].*','', text)

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

with open('HP-german_clean.txt', 'w', encoding='utf-8-sig') as f:
    f.write(text)