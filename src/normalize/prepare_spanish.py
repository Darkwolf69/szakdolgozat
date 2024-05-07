# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 01:38:56 2023

@author: adamWolf

Prepare french text to additional normalizing
'''

def prepare_spanish(text):

    #remove unnecessary em-dash characters
    prepared_text = text.replace('—','')
    return prepared_text
