# -*- coding: utf-8 -*-
'''
Created on Wed Nov 15 22:58:32 2023

@author: adamWolf
'''

import basic_functions
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-french.txt')).read()

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

with open('HP-french_clean.txt', 'w', encoding='utf-8-sig') as f:
    f.write(text)