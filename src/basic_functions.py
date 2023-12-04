# -*- coding: utf-8 -*-
'''
Created on Mon Oct 16 21:50:22 2023

@author: adamWolf
'''

def open_text(file_name):
    opened_file = open(file_name, 'r', encoding='utf-8-sig')
    return opened_file

def close_text(file_name):
    file_name.close()
    
def write_text(file_name, new_text):
    with open(file_name, 'w', encoding='utf-8-sig') as f:
        f.write(new_text)

def line_number_of_text(text):
    number_of_lines = text.count('\n') + 1
    return number_of_lines

def number_of_words_in_text(text):
    number_of_words = 0
    lines = text.split()
    number_of_words += len(lines)
    return number_of_words