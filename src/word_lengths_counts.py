# -*- coding: utf-8 -*-
'''
Created on Sat Nov 11 21:18:27 2023

@author: adamWolf
'''

import matplotlib.pyplot as plt

def process_text_for_lengths(text, language):
    """
    Processes the text to count word lengths and find the longest words.
    
    Parameters:
        text (str): The input text to be analyzed.
        language (str): The language of the text.

    Returns:
        dict: A dictionary with word lengths as keys and their counts as values.
        list: A list of the four longest words in the text.
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        words = [x for x in text.split() if x]
        for suffix in ('’', '”'):
            words = [x.removesuffix(suffix) for x in words]
        for prefix in ('“', '‘'):
            words = [x.removeprefix(prefix) for x in words]
        if language == 'english':
            words = list(filter(lambda x: x != "'", words))
        words = list(filter(None, words))
        
        dictionary = {}
        for x in words:
            word_length = len(x)
            dictionary[word_length] = dictionary.get(word_length, 0) + 1
        
        longests = sorted(words, key=len)[-4:]
        
    return dictionary, longests

def word_lengths_counts(text, language):
    """
    Analyzes the distribution of word lengths in the provided text and generates a bar plot 
    showing the counts of words of each length. Also, displays the four longest words in the text.

    Parameters:
        text (str): The input text to be analyzed.
        language (str): The language of the text.

    Returns:
        matplotlib.pyplot: A bar plot showing the distribution of word lengths and the titles
        indicating the four longest words in the text.

    Raises:
        AttributeError: If the input text is not a string.
    """
    dictionary, longests = process_text_for_lengths(text, language)
    
    lengths, counts = zip(*dictionary.items())
    plt.figure(figsize=(14, 7))
    plt.bar(lengths, counts)
    plt.xticks(range(1, max(lengths) + 1))
    plt.xlabel('Word lengths')
    plt.ylabel('Word counts')
    plt.title(f'The 4 longest words in {language} text:\n' + '\n'.join(reversed(longests)), wrap=True)
    plt.show()

def doctest_word_lengths_counts():
    """
    Tests the process_text_for_lengths function with sample data.

    Examples:
        >>> process_text_for_lengths("This is a test sentence with hyphenated words like well-known and Hagrid’s speech.", "english")
        ({4: 4, 2: 1, 1: 1, 8: 2, 10: 2, 5: 1, 3: 1, 7: 1}, ['sentence', 'Hagrid’s', 'hyphenated', 'well-known'])

        >>> process_text_for_lengths("Este es un texto de prueba con palabras en español.", "spanish")
        ({4: 1, 2: 4, 5: 1, 6: 1, 3: 1, 8: 2}, ['texto', 'prueba', 'palabras', 'español.'])

        >>> process_text_for_lengths(12345, "english")
        Traceback (most recent call last):
        AttributeError: Invalid input
    """
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
