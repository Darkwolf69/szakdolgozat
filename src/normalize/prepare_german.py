# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:02:34 2023

@author: adamWolf

Prepare german text to additional normalizing
'''

import re


def prepare_german(text):
    
    #remove unnecessary footer texts and page numbers
    prepared_text = re.sub('HARRY POTTER [a-z].*','', text)
    
    return(prepared_text)