# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:38:56 2023

@author: adamWolf
'''

import basic_functions
import re
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-spanish.txt')).read()

#remove unnecessary chapter words
text = re.sub('CAPÍTULO [A-Z]+','', text)
text = re.sub('CAPÍTULO [0-9]+','', text)

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

with open('HP-spanish_clean.txt', 'w', encoding='utf-8-sig') as f:
    f.write(text)
