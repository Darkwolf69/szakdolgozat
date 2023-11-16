# -*- coding: utf-8 -*-
'''
Created on Thu Nov 16 00:05:20 2023

@author: adamWolf
'''

import basic_functions
import matplotlib.pyplot as plt


### Count and display the
data = basic_functions.open_text('HP-french_clean.txt').read()

# split data into a list of words
words = [x for x in data.split() if x]

#filter remaining empty strings
words = list(filter(None, words))
print(words)

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

# what is the longest word
plt.title('Longest word in HP-french: ' + ' '.join(x for x in words if len(x)==max(lengths)))

plt.show()

clear_file = ' '.join(words)
with open('HP-french_final.txt', 'w', encoding='utf-8-sig') as f:
    f.write(clear_file)