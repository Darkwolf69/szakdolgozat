# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:39:29 2023

@author: adamWolf
"""

def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    text = '\n'.join(non_empty_lines)
    return text