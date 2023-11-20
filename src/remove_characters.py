# -*- coding: utf-8 -*-
'''
Created on Mon Nov 13 02:13:27 2023

@author: adamWolf
'''

import re

def remove_unnecessary_chars(data):

    # remove unnecessary characters from text data
    #(¿¡ are only in spanish)
    data = re.sub('[,.:;!?"()–*»«…¿¡]','',data)
    data = data.replace(' – ',' ')
    data = data.replace('...',' ')

    return data