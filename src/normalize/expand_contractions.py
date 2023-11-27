# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:52:32 2023

@author: adamWolf

works only with less than a specific length of string
"""

import contractions

def expand_contractions(text):
    return contractions.fix(text)
