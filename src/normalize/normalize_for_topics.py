# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:53:16 2023

@author: adamWolf
"""

import basic_functions
import remove_empty_lines
import remove_stopwords
import os
import re
import nltk
import gensim
import numpy as np


text = (basic_functions.open_text('HP-english.txt')).read()

#some normalization
#remove hyphenations and unify hyphenated words
text = re.sub(r'-\n(\w+ *)', r'\1\n', text)
text = re.sub(r'-\n\n(\w+ *)', r'\1\n', text)
#remove unnecessary em-dash, header and footer texts, page numbers and chapter words
text = re.sub('[0-9]+ HARRY  POTTER','', text)
text = re.sub('.*[A-Z] [0-9]+','', text)
text = re.sub('[—]','',text)
#remove empty lines
text = remove_empty_lines.remove_empty_lines(text)



#make chapters
num_of_chapters = 17

#generate folders for storing chapter files in all languages
if not os.path.exists('C:\\Users\\farka\\Harry_Potter\\normalize\\chapters_for_topics\\'):
    os.makedirs('C:\\Users\\farka\\Harry_Potter\\normalize\\chapters_for_topics\\')

# Prefix extraction before specific string (removes 'chapter xy' words from chapters)
for i in range(0, num_of_chapters):
    res = text.rsplit('CHAPTER', 16)[i]
    res2 = res.split(' ', 2)[2]
    if i < 9:
        subdir = 'chapters_for_topics'
        filename = f'HP_chapter_0{i+1}.txt'
        path = os.path.join(subdir, filename)
        with open(path, 'w', encoding='utf-8-sig') as f: 
            f.write(res2)
    else:
        subdir = 'chapters_for_topics'
        filename = f'HP_chapter_{i+1}.txt'
        path = os.path.join(subdir, filename)
        with open(path, 'w', encoding='utf-8-sig') as f: 
            f.write(res2)

#load chapters into a list
chapters = []
file_path = 'C:\\Users\\farka\\Harry_Potter\\normalize\\chapters_for_topics\\'
file_names = os.listdir(file_path)

for file_name in file_names:
    file_path = f'C:\\Users\\farka\\Harry_Potter\\normalize\\chapters_for_topics\\{file_name}'
    with open(file_path, encoding='utf-8', errors='ignore', mode='r+') as f:
        data = f.read()
        chapters.append(data)



#normalization of chapters
stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

#removes letter 's' from the end of 'Mrs' in Mrs Dursley?
def normalize_corpus(papers):
    norm_papers = []
    for paper in papers:
        #TODO: 'wa' words remain in text for some reason?
        remove_stopwords.remove_stopwords(paper, 'english')
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = list(filter(None, paper_tokens))
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers
    
norm_papers = normalize_corpus(chapters)
# print(len(norm_papers))
# print(norm_papers)
# print('\n')



#handling bi-grams
bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20,
delimiter='_') # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)
# sample demonstration
# print(bigram_model[norm_papers[6]][:50])
# print('\n')



#generate phrases for all our tokenized chapters and build a vocabulary
#for creating a unique term/phrase to number mapping
norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]
# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
# print('Sample word to number mappings:', list(dictionary.items())[:15])
# print('\n')
print('\nTotal Vocabulary Size:', len(dictionary))

# Filter out words that occur less than 2 chapters, or more than 50% of the chapters
# this filters out recurring topics
dictionary.filter_extremes(no_below=6, no_above=0.8)
print('Total Filtered Vocabulary Size:', len(dictionary))




#feature engineering with a bag of words model
# Transforming corpus into bag of words vectors
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
# print(bow_corpus[1][:50])
# # viewing actual terms and their counts
# print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])
# total chapters in the corpus
print('\nTotal number of chapters:', len(bow_corpus))


#latent semantic indexing
TOTAL_TOPICS = 5
lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary,
                                 num_topics=TOTAL_TOPICS,
                                 onepass=True,
                                 chunksize=1740, 
                                 power_iters=1000)

# =============================================================================
# latent semantic indexing topics
# for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
#     print('Topic #'+str(topic_id+1)+':')
#     print(topic)
#     print()
# =============================================================================

#separate the positive and the negative weights
# for n in range(TOTAL_TOPICS):
#     print('Topic #'+str(n+1)+':')
#     print('='*50)
#     d1 = []
#     d2 = []
#     for term, wt in lsi_bow.show_topic(n, topn=20):
#         if wt >= 0:
#             d1.append((term, round(wt, 3)))
#         else:
#             d2.append((term, round(wt, 3)))
#     print('Direction 1:', d1)
#     print('-'*50)
#     print('Direction 2:', d2)
#     print('-'*50)
#     print()



#Latent Dirichlet Allocation (LDA)
lda_model = gensim.models.LdaModel( corpus=bow_corpus, id2word=dictionary,
                                   chunksize=1740, alpha='auto',
                                   eta='auto', random_state=42,
                                   iterations=500, num_topics=TOTAL_TOPICS,
                                   passes=20, eval_every=None)

# for topic_id, topic in lda_model.print_topics(num_topics=5, num_words=50):
#     print('Topic #'+str(topic_id+1)+':')
#     print(topic)
#     print()

#overall mean coherence score of the model
topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
avg_coherence_score = np.mean([item[1] for item in topics_coherences])
print('\nAvg. Coherence Score:', avg_coherence_score)

#LDA topics with weights
topics_with_wts = [item[0] for item in topics_coherences]
# print('LDA Topics with Weights')
# print('='*50)
# for idx, topic in enumerate(topics_with_wts):
#     print('Topic #'+str(idx+1)+':')
#     print([(term, round(wt, 3)) for wt, term in topic])
#     print()
    
#LDA topics without weights
print('\nLDA Topics without Weights')
print(f'Total topics: {TOTAL_TOPICS}')
print('='*50)
for idx, topic in enumerate(topics_with_wts):
    print('Topic #'+str(idx+1)+':')
    print([term for wt, term in topic])
    print()
    
    
## not working yet
# cv_coherence_model_lda = gensim.models.CoherenceModel( model=lda_model,
#                                                       corpus=bow_corpus,
#                                                       texts=norm_corpus_bigrams,
#                                                       dictionary=dictionary,
#                                                       coherence='c_v')
# avg_coherence_cv = cv_coherence_model_lda.get_coherence()

# umass_coherence_model_lda = gensim.models.CoherenceModel( model=lda_model,
#                                                          corpus=bow_corpus,
#                                                          texts=norm_corpus_bigrams,
#                                                          dictionary=dictionary,
#                                                          coherence='u_mass')
# avg_coherence_umass = umass_coherence_model_lda.get_coherence()

# perplexity = lda_model.log_perplexity(bow_corpus)

# print('Avg. Coherence Score (Cv):', avg_coherence_cv)
# print('Avg. Coherence Score (UMass):', avg_coherence_umass)
# print('Model Perplexity:', perplexity)
