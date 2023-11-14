# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:18:27 2023

@author: farka
"""

import basic_functions
import remove_characters as remove_chars
import matplotlib.pyplot as plt


### Count and display the
data = (basic_functions.open_text("HP-english_clean.txt")).read()

data = remove_chars.remove_unnecessary_chars(data)

# split data into a list of words
words = [x for x in data.split() if x]

# remove suffixes and prefixes
for suffix in ('’','”'):
    words = [x.removesuffix(suffix) for x in words]
    
for prefix in ('“','‘'):
    words = [x.removeprefix(prefix) for x in words]

#filter remaining empty strings
words = list(filter(None, words))

clear_file = ' '.join(words)
with open("HP-english_clean.txt", "w", encoding="utf-8-sig") as f:
    f.write(clear_file)
    
# count the different lenghts using a dict
dictionary = {}
for x in words:
    word_length = len(x)
    dictionary[word_length] = dictionary.get(word_length, 0) + 1

# retrieve the relevant info from the dict 
lengths, counts = zip(*dictionary.items())

# plot the relevant info
plt.bar(lengths, counts)
plt.xticks(range(1, max(lengths)+1))
plt.xlabel('Word lengths')
plt.ylabel('Word counts')

# what is the longest word?
plt.title(' '.join(w for w in words if len(w)==max(lengths)))

plt.show()
