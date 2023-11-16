# -*- coding: utf-8 -*-
'''
Created on Sun Nov 12 23:51:01 2023

@author: adamWolf
'''

import basic_functions
import re
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-english.txt')).read()

#remove hyphenations and unify hyphenated words
text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)

#remove unnecessary header and footer texts, page numbers and chapter words
text = re.sub('[0-9]+ HARRY  POTTER','', text)
text = re.sub('.*[A-Z] [0-9]+','', text)
text = re.sub('— CHAPTER [A-Z]+ —','', text)

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

with open('HP-english_clean.txt', 'w', encoding='utf-8-sig') as f:
    f.write(text)