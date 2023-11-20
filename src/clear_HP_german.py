# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:02:34 2023

@author: adamWolf

Open and clear original german text file, to prepare for statistic analysis.
Output is a text containing just words including chapter titles and newline characters.
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

basic_functions.write_text('HP-german_clean.txt', text)