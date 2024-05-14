# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:47:53 2023

@author: adamWolf
"""

import re

import basic_functions
import remove_characters
import remove_empty_lines


def clear_texts_to_chapters():
    """
    Read texts from files, clean and preprocess them, and write chapter-wise versions for different languages.

    Reads the text from the 'HP-english.txt' file, removes hyphenations and unnecessary characters, splits
    the text into words, removes suffixes and prefixes, and removes empty lines. Writes the cleaned English text
    to 'HP-english_to_chapters.txt'. 

    Also processes German, French, and Spanish texts similarly, removing newline characters and writing the
    processed texts to 'HP-german_to_chapters.txt', 'HP-french_to_chapters.txt', and 'HP-spanish_to_chapters.txt'
    respectively.

    Note:
        This function uses helper functions 'basic_functions.open_text', 'remove_characters.remove_unnecessary_chars',
        'remove_empty_lines.remove_empty_lines', and 'basic_functions.write_text'.

    Raises:
        IOError: If there's an issue with reading from or writing to a file.
    """
    try:
        text = (basic_functions.open_text('HP-english.txt')).read()
    except IOError:
        print("Something went wrong when reading from the file")
    else:
        #remove hyphenations and unify hyphenated words
        text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
        text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)
        
        #remove unnecessary em-dash, header and footer texts, page numbers and chapter words
        text = re.sub('[0-9]+ HARRY  POTTER', '', text)
        text = re.sub('.*[A-Z] [0-9]+', '', text)
        text = re.sub('[—]', '',text)
        
        #remove unnecessary characters (dots, etc.)
        text = remove_characters.remove_unnecessary_chars(text)
        
        # split data into a list of words
        words = [x for x in text.split() if x]
        # remove suffixes and prefixes
        for suffix in ('’', '”'):
            words = [x.removesuffix(suffix) for x in words]
        for prefix in ('“', '‘'):
            words = [x.removeprefix(prefix) for x in words]
        clean_text = ' '.join(words)
        
        #remove empty lines
        text = remove_empty_lines.remove_empty_lines(text)
        
        try:
            basic_functions.write_text('HP-english_to_chapters.txt', clean_text)
            
            #remove newline characters from texts (except for english)
            text = (basic_functions.open_text('HP-german_clean.txt')).read()
            text = re.sub('[\n]', ' ', text)
            basic_functions.write_text('HP-german_to_chapters.txt', text)
            
            #unnecessary, since chapter texts and titles are in the same line, but done for integrity
            text = (basic_functions.open_text('HP-french_clean.txt')).read()
            text = re.sub('[\n]', ' ', text)
            basic_functions.write_text('HP-french_to_chapters.txt', text)
            
            text = (basic_functions.open_text('HP-spanish_clean.txt')).read()
            text = re.sub('[\n]', ' ' ,text)
            basic_functions.write_text('HP-spanish_to_chapters.txt', text)
        except IOError:
            print("Something went wrong when writing to the file")
        finally:
            text.close()
            

clear_texts_to_chapters()
        