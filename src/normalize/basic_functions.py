# -*- coding: utf-8 -*-
'''
Created on Mon Oct 16 21:50:22 2023

@author: adamWolf

basic IO and statistic functions
'''

def open_text(file_name):
    try:
        opened_file = open(file_name, 'r', encoding='utf-8-sig')
    except IOError:
        print("Something went wrong when reading from the file")
    else:
        return opened_file
    
    
def close_text(file_name):
    try:
        file_name.close()
    except IOError:
        print("Something went wrong when closing the file")
    
    
def write_text(file_name, new_text):
    with open(file_name, 'w', encoding='utf-8-sig') as f:
        try:
            f.write(new_text)
        except IOError:
            print("Something went wrong when writing to the file")


def line_number_of_text(text):
    """
    Calculates the number of lines in the provided text.

    Parameters:
        text (str): The input text for which line count is to be determined.
 
    Returns:
        int: The number of lines in the text.
 
    Raises:
        AttributeError: If the input is not a string.
        
    
    >>> line_number_of_text("This is a single line.")
    1
    >>> line_number_of_text("This is the first line.\\nThis is the second line.")
    2
    >>> line_number_of_text("Line one.\\nLine two.\\nLine three.")
    3
    >>> line_number_of_text("")
    1
    >>> line_number_of_text(123)
    Traceback (most recent call last):
    AttributeError: Invalid input
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        number_of_lines = text.count('\n') + 1
        return number_of_lines


def number_of_words_in_text(text):
    """
    Calculates the number of words in the provided text.

    Parameters:
        text (str): The input text for which word count is to be determined.

    Returns:
        int: The number of words in the text.

    Raises:
        AttributeError: If the input is not a string.
        
    
    >>> number_of_words_in_text("This is a test.")
    4
    >>> number_of_words_in_text("One more example with a few words.")
    7
    >>> number_of_words_in_text("")
    0
    >>> number_of_words_in_text("SingleWord")
    1
    >>> number_of_words_in_text(123)
    Traceback (most recent call last):
    AttributeError: Invalid input
    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        number_of_words = 0
        words = text.split()
        number_of_words += len(words)
        return number_of_words


if __name__ == "__main__":
    import doctest
    doctest.testmod()