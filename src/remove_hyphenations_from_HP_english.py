# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:51:01 2023

@author: farka
"""

import basic_functions
import re

text = (basic_functions.open_text("HP-english.txt")).read()

text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)

with open("HP-english_clean.txt", "w", encoding="utf-8") as f:
    f.write(text)