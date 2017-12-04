
# coding: utf-8

# In[1]:


import numpy as np
from __future__ import division

filename = 'glove.6B.100d.txt'

def loadEmbeddings(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded!')
    file.close()
    return vocab,embd
vocab,embd = loadEmbeddings(filename)


word_vec_dim = len(embd[0])

vocab.append('<UNK>')
embd.append(np.asarray(embd[vocab.index('unk')],np.float32)+0.01)

vocab.append('<EOS>')
embd.append(np.asarray(embd[vocab.index('eos')],np.float32)+0.01)

vocab.append('<PAD>')
embd.append(np.zeros((word_vec_dim),np.float32))

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('<UNK>')]

def most_similar_eucli(x):
    xminusy = np.subtract(embedding,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)

def vec2word(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli(np.asarray(vec,np.float32))
    return vocab[most_similars[0]]


# In[2]:


import csv

    
sentences=[]
tags=[]
NER_tags = []
max_len = 30

with open('ner.csv') as csvfile:
    
    reader = csv.DictReader(csvfile)
    
    sentence=[]
    tag=[]
    i=0
    
    for row in reader:
        if row['Sentence #']!='' and i!=0:
            
            if len(sentence) <= max_len:
                sentences.append(sentence)
                tags.append(tag)
            sentence=[]
            tag=[]
            
        sentence.append(row['Word'].lower())
        
        temp = row['Tag']

        if temp not in NER_tags:
            NER_tags.append(temp)
        
        tag.append(NER_tags.index(temp))
        
        i+=1

    print "No. of samples: "+str(len(sentences))


# In[3]:


print NER_tags


# In[4]:


enough = 10000

sentences = sentences[0:enough]
tags = tags[0:enough]


# In[5]:


vectorized_sentences = []

for sentence in sentences:
    vectorized_sentences.append(map(word2vec,sentence))


# In[6]:


padded_sentences = np.zeros((len(sentences),max_len,word_vec_dim),np.float32)
padded_tags = np.zeros((len(tags),max_len),np.int32)

pad_word = np.zeros((word_vec_dim),np.float32)
pad_tag = NER_tags.index('O')

for i in xrange(len(sentences)):
    
    for j in xrange(max_len):
        
        if j >= len(sentences[i]):
            padded_sentences[i,j] = pad_word
            padded_tags[i,j] = pad_tag
        else:
            padded_sentences[i,j] = vectorized_sentences[i][j]
            padded_tags[i,j] = tags[i][j]


# In[7]:


#Saving processed data in another file.

import pickle

PICK = [NER_tags,padded_sentences,padded_tags]

with open('NERPICKLE_10000', 'wb') as fp:
    pickle.dump(PICK, fp)

