# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:12:56 2023

@author: adamWolf

remove all unnecessary characters from text data which are not
ASCII characters (whitelist)
"""

import re


def remove_special_characters(text, remove_digits=False):
    
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    
    return text