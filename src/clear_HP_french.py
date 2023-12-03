# -*- coding: utf-8 -*-
'''
Created on Wed Nov 15 22:58:32 2023

@author: adamWolf

Open and clear original french text file, to prepare for statistic analysis.
Output is a text containing just words including chapter titles and newline characters.
'''

import basic_functions
import re
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-french.txt')).read()

# unify hyphens (change EM-DASHes to HYPHEN-MINUS)
text = re.sub('—', '-', text)

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)

basic_functions.write_text('HP-french_clean.txt', text)