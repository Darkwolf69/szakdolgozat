# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:00:39 2023

@author: adamWolf

Performs summarization of chapters and the entire book using different techniques.
This function summarizes each chapter of the book "Harry Potter And The Philosopher's Stone"
using Latent Semantic Indexing with Singular Value Decomposition (LSI_SVD) and TextRank algorithms.
It also summarizes the entire book using LSI_SVD and TextRank.

Returns:
    None (writes the summarization results to files)
"""

import os

import summarization_LSI_SVD as summ_LSI_SVD
import summarization_text_rank as sum_text_rank

import basic_functions
import prepare_english as prep_eng


actual_dir = os.path.dirname(os.path.realpath('__file__'))

# summarize each chapter of the book with LSI_SVD
num_of_texts = 17

num_of_sentences_chapters = 40
num_of_topics_chapters = 3
          
with open(f'summarization_LSI_SVD_chapters_{num_of_topics_chapters}_topic.txt', 'w', encoding = 'utf-8-sig') as f:
    f.write('Harry Potter And The Philospohers Stone\n'
            'Summarization of the whole book with LSI_SVD\n'
            f'Number of sentences in each chapter: {num_of_sentences_chapters}\n'
            f'Number of topics: {num_of_topics_chapters}\n\n')
for i in range(0, num_of_texts):
    if (i < 9):
        file_name = os.path.join(actual_dir, f'HP-english_chapters\HP-english_chapter_0{i+1}.txt')
    else:
        file_name = os.path.join(actual_dir, f'HP-english_chapters\HP-english_chapter_{i+1}.txt')
    
    text = open(file_name, 'r', encoding='utf-8-sig').read()
    sum_text = summ_LSI_SVD.summarize_LSI_SVD(text, num_of_sentences_chapters, num_of_topics_chapters)
    with open(f'summarization_LSI_SVD_chapters_{num_of_topics_chapters}_topic.txt', 'a', encoding = 'utf-8-sig') as f:
          f.write('\n\n##############################################\n'
                  f'      Summarization of CHAPTER {i+1}\n'
                  '##############################################\n\n'
                  f'\n{sum_text}')



# summarize each chapter of the book with TextRank
with open('summarization_text_rank_chapters.txt', 'w', encoding = 'utf-8-sig') as f:
      f.write('Harry Potter And The Philospohers Stone\n'
                'Summarization of the whole book with TextRank\n'
                f'Number of sentences in each chapter: {num_of_sentences_chapters}\n')
              
for i in range(0, num_of_texts):
    if (i < 9):
        file_name = os.path.join(actual_dir, f'HP-english_chapters\HP-english_chapter_0{i+1}.txt')
    else:
        file_name = os.path.join(actual_dir, f'HP-english_chapters\HP-english_chapter_{i+1}.txt')
    
    text = open(file_name, 'r', encoding='utf-8-sig').read()
    sum_text = sum_text_rank.summarize_text_rank(text, num_of_sentences_chapters)
    with open('summarization_text_rank_chapters.txt', 'a', encoding = 'utf-8-sig') as f:
          f.write('\n\n##############################################\n'
                    '   Harry Potter And The Philospohers Stone   \n'
                  f'      Summarization of CHAPTER {i+1}\n'
                  '##############################################\n\n'
                  f'\n{sum_text}')
    


whole_eng_text = (basic_functions.open_text('HP-english.txt')).read()
# summarize the whole book with LDA
whole_eng_text = prep_eng.prepare_english(whole_eng_text)
number_of_sentences = 500
number_of_topics = 10

sum_text_LSI_SVD = summ_LSI_SVD.summarize_LSI_SVD(whole_eng_text,
                                                  number_of_sentences,
                                                  number_of_topics)
with open(f'summarization_LSI_SVD_{number_of_topics}_topic.txt', 'w', encoding = 'utf-8-sig') as f:
    f.write('Harry Potter And The Philospohers Stone\n'
            'Summarization of the whole book with LSI_SVD\n'
            'Number of sentences: 500\n'
            f'Number of topics: {number_of_topics}\n\n'
            f'{sum_text_LSI_SVD}')

# summarize the whole book with TextRank
sum_text_rank = sum_text_rank.summarize_text_rank(whole_eng_text, number_of_sentences)
with open('summarization_text_rank.txt', 'w', encoding = 'utf-8-sig') as f:
    f.write('Harry Potter And The Philospohers Stone\n'
            'Summarization of the whole book with TextRank\n'
            f'Number of sentences: {number_of_sentences}\n\n{sum_text_rank}')


