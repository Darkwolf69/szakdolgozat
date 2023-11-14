# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 02:13:27 2023

@author: farka
"""

import re

def remove_unnecessary_chars(data):

    # remove unnecessary characters from text data
    data = re.sub('[,.:;!?"()–—]','',data)
    data = data.replace(' – ',' ')

    return data
