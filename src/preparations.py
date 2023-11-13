# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:46:10 2023

@author: farka
"""

import basic_functions as base_fns
import remove_hyphenations_from_HP_english
import remove_characters as rm_chars

remove_hyphenations_from_HP_english
rm_chars.remove_unnecessary_chars((base_fns.open_text("HP-english.txt")).read())
