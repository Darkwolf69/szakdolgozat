# -*- coding: utf-8 -*-
'''
Created on Mon Nov 13 02:13:27 2023

@author: adamWolf
'''

import re


def remove_unnecessary_chars(text):
    """
    Removes unnecessary characters from the given text in all 4 languages,
    but keeps non-ASCII letters (blacklist)

    Parameters:
        text (str): The input text.

    Returns:
        str: The text with unnecessary characters removed.

    Raises:
        AttributeError: If the input is not a string.

    """
    if type(text) is not str:
        raise AttributeError('Invalid input')
    else:
        #(¿¡ occur only in spanish text)
        text = re.sub('[,.:;!?"()–*»«…¿¡]', '', text)
        text = text.replace(' – ', ' ')
        text = text.replace('...', ' ')

    return text