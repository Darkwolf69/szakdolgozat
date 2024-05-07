# -*- coding: utf-8 -*-
'''
Created on Wed Nov 15 22:58:32 2023

@author: adamWolf

Prepare french text to additional normalizing
'''

def prepare_french(text):
    
    # unify hyphens (replace EM-DASH-es with HYPHEN-MINUS-es)
    prepared_text = text.replace('—', '-')
    
    return(prepared_text)

