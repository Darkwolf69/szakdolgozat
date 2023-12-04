# -*- coding: utf-8 -*-
'''
Created on Mon Nov 13 02:13:27 2023

@author: adamWolf

remove unnecessary characters from text data in all 4 languages
but keep non-ASCII letters (blacklist)
'''

import re

def remove_unnecessary_chars(text):
    #(¿¡ are only in spanish text)
    text = re.sub('[,.:;!?"()–*»«…¿¡]','',text)
    text = text.replace(' – ',' ')
    text = text.replace('...',' ')

    return text