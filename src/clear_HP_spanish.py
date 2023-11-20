# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:38:56 2023

@author: adamWolf

Open and clear original spanish text file, to prepare for statistic analysis.
Output is a text containing just words including chapter titles and newline characters.
'''

import basic_functions
import remove_characters
import remove_empty_lines


text = (basic_functions.open_text('HP-spanish.txt')).read()

#remove unnecessary em-dash characters
text = text.replace('—','')

#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)

#remove unnecessary characters (dots, etc.)
text = remove_characters.remove_unnecessary_chars(text)


basic_functions.write_text('HP-spanish_clean.txt', text)
