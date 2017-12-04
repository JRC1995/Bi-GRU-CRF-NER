
# coding: utf-8

# ### Load Word Vectors....
# 
# and define functions to convert words to vectors and vice-versa.

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


# ### Load Pre-processed Data
# 
# (Minimally processed data - only used pre-trained GloVe and padding to a fixed length)

# In[2]:


import pickle
with open ('NERPICKLE_10000', 'rb') as fp:
    processed_data = pickle.load(fp)

NER_tags = processed_data[0]
sentences = processed_data[1]
tags = processed_data[2]


# ### Splitting into training, validating and testing data.

# In[3]:


p = 0.8

train_len = int(.8*len(sentences))
val_len = int(.1*len(sentences))
test_len = len(sentences) - train_len - val_len

train_sentences = sentences[0:train_len]
train_tags = tags[0:train_len]

val_sentences = sentences[train_len:train_len+val_len]
val_tags = tags[train_len:train_len+val_len]

test_sentences = sentences[train_len+val_len:]
test_tags = tags[train_len+val_len:]


# ### Function to create 'meta-batches' of sentences and the target NER tags

# In[4]:


def create_batches(sentences,tags,batch_size):
    
    shuffle = np.arange(len(sentences))
    np.random.shuffle(shuffle)
    
    batches_sentence = []
    batches_tag = []
    
    i=0
    
    while i+batch_size<=len(sentences):
        
        batch_sentence = []
        batch_tag = []

        for j in xrange(i,i+batch_size):
            
            batch_sentence.append(sentences[shuffle[j]])
            batch_tag.append(tags[shuffle[j]])
            
        batch_sentence = np.asarray(batch_sentence,np.float32)
        batch_tag = np.asarray(batch_tag,np.int32)
        
        batches_sentence.append(batch_sentence)
        batches_tag.append(batch_tag)

        i+=batch_size
        
    batches_sentence = np.asarray(batches_sentence,np.float32)
    batches_tag = np.asarray(batches_tag,np.int32)

    return batches_sentence,batches_tag
    


# ### Hyperparamters

# In[5]:


import tensorflow as tf

# Hyperparameters

tf_sentences = tf.placeholder(tf.float32,[None,None,word_vec_dim])
tf_tags = tf.placeholder(tf.int32,[None,None])
traintestval = tf.placeholder(tf.bool)
scale_down = 1

hidden_size = 100
learning_rate = 0.01
beta = 0.0001
regularizer = tf.contrib.layers.l2_regularizer(scale=beta)


# ### Parameters

# In[6]:


# Parameters
init = tf.zeros_initializer()
    
    
with tf.variable_scope("Bi_GRU"):

    # FORWARD GRU PARAMETERS
    
    wzf = tf.get_variable("wzf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer= regularizer)
    uzf = tf.get_variable("uzf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzf = tf.get_variable("bzf", shape=[hidden_size],initializer=init)

    wrf = tf.get_variable("wrf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urf = tf.get_variable("urf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brf = tf.get_variable("brf", shape=[hidden_size],initializer=init)

    wf = tf.get_variable("wf", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    uf = tf.get_variable("uf", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bf = tf.get_variable("bf", shape=[hidden_size],initializer=init)

    # BACKWARD GRU PARAMETERS

    wzb = tf.get_variable("wzb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    uzb = tf.get_variable("uzb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzb = tf.get_variable("bzb", shape=[hidden_size],initializer=init)

    wrb = tf.get_variable("wrb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urb = tf.get_variable("urb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brb = tf.get_variable("brb", shape=[hidden_size],initializer=init)

    wb = tf.get_variable("wb", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    ub = tf.get_variable("ub", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bb = tf.get_variable("bb", shape=[hidden_size],initializer=init)
    

W_score = tf.get_variable("W_score", shape=[2*hidden_size,len(NER_tags)],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)

B_score = tf.get_variable("B_score", shape=[len(NER_tags)],initializer=init)

Transition_matrix = tf.get_variable("T",shape=[len(NER_tags),len(NER_tags)],
                                    initializer=tf.random_normal_initializer(),
                                    regularizer=regularizer)

l1 = tf.get_variable("l1", shape=[1],
                     initializer=tf.constant_initializer(0.5),
                     regularizer=regularizer)

l2 = tf.get_variable("l2", shape=[1],
                     initializer=tf.constant_initializer(0.5),
                     regularizer=regularizer)
    


# ### Function for layer normalization
# 
# Without scale and shift.

# In[7]:


def layer_norm(inputs,scope,epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = tf.multiply((1/ tf.sqrt(var + epsilon)),(inputs - mean))
 
    return LN


# ### Barebones implementation of Bi-directional GRU

# In[8]:


def bi_GRU(inp,hidden,seq_len,scope):
    
    #inp shape = batch_size x seq_len x vector_dimension
    
    inp = tf.transpose(inp,[1,0,2])
    
    #now inp shape = seq_len x batch_size x vector_dimension
    
    hidden_forward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    hidden_backward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hiddenf = hidden
    hiddenb = hidden

    with tf.variable_scope(scope, reuse=True):
        
        wzf = tf.get_variable("wzf")
        uzf = tf.get_variable("uzf")
        bzf = tf.get_variable("bzf")
        
        wrf = tf.get_variable("wrf")
        urf = tf.get_variable("urf")
        brf = tf.get_variable("brf")
        
        wf = tf.get_variable("wf")
        uf = tf.get_variable("uf")
        bf = tf.get_variable("bf")
        
        wzb = tf.get_variable("wzb")
        uzb = tf.get_variable("uzb")
        bzb = tf.get_variable("bzb")
        
        wrb = tf.get_variable("wrb")
        urb = tf.get_variable("urb")
        brb = tf.get_variable("brb")
        
        wb = tf.get_variable("wb")
        ub = tf.get_variable("ub")
        bb = tf.get_variable("bb")
        
    i = 0
    j = seq_len - 1
    
    def cond(i,j,hiddenf,hiddenb,hidden_forward,hidden_backward):
        return i < seq_len
    
    def body(i,j,hiddenf,hiddenb,hidden_forward,hidden_backward):
        
        xf = inp[i]
        xb = inp[j]

        # FORWARD GRU EQUATIONS:
        z = tf.sigmoid( tf.matmul(xf,wzf) + tf.matmul(hiddenf,uzf) + bzf )
        r = tf.sigmoid( tf.matmul(xf,wrf) + tf.matmul(hiddenf,urf) + brf )
        h_ = tf.tanh( tf.matmul(xf,wf) + tf.multiply(r,tf.matmul(hiddenf,uf)) + bf )
        hiddenf = tf.multiply(z,h_) + tf.multiply((1-z),hiddenf)

        hidden_forward = hidden_forward.write(i,hiddenf)
        
        # BACKWARD GRU EQUATIONS:
        z = tf.sigmoid( tf.matmul(xb,wzb) + tf.matmul(hiddenb,uzb) + bzb )
        r = tf.sigmoid( tf.matmul(xb,wrb) + tf.matmul(hiddenb,urb) + brb )
        h_ = tf.tanh( tf.matmul(xb,wb) + tf.multiply(r,tf.matmul(hiddenb,ub)) + bb )
        hiddenb = tf.multiply(z,h_) + tf.multiply((1-z),hiddenb)
        
        hidden_backward = hidden_backward.write(j,hiddenb)
        
        
        return i+1,j-1,hiddenf,hiddenb,hidden_forward,hidden_backward
    
    _,_,_,_,hidden_forward,hidden_backward = tf.while_loop(cond,body,[i,j,
                                                                        hiddenf,
                                                                        hiddenb,
                                                                        hidden_forward,
                                                                        hidden_backward])
    
    forward = hidden_forward.stack()
    backward = hidden_backward.stack()
    
    hidden_list = tf.concat([forward,backward],2)
    
    #forward\backward\hidden_list shape = seq_len x  batch_size x 2*hidden_size
    
    hidden_list = tf.transpose(hidden_list,[1,0,2])
    
    #now hidden_list shape = batch_size x seq_len x 2*hidden_size
    
    return hidden_list
    

        


# ### The model
# 
# The fist part (The GRU layer) is straight forward. The input sentences are passed through a bi-directional GRU and then each words are transformed to a unnormalized probability distribution over the NER tags through a linear layer.
# 
# The second part is a CRF layer. The following is the core equation of CRF. 

# 
# \begin{equation}
# P(y_{1:N}\mid x_{1:N})  =\frac1Zexp( \Sigma^N_{n=1} \Sigma^F_{i=1} \lambda_i f(y_{n-1},y_n,x_n)
# \end{equation}
# 

# Here P(y1:n | x1: n) is the probability of the sequence y1,y2....yn given the inputs x1,x2,...xn.
# Z is the normalizing factor. 

# To train the CRF negative log likelihood of the P(y|x) can be used as the cost function, where y is the target sequence, and x is the input sequence. The equation of the cost function can be thus expressed as:-
# 
# \begin{align}
# -log(P(y_{1:N}\mid x_{1:N})) &= -log(\frac1Zexp( \Sigma^N_{n=1} \Sigma^F_{i=1} \lambda_i f(y_{n-1},y_n,x_n))\\
#                              &= -(\Sigma^N_{n=1} \Sigma^F_{i=1} \lambda_i f(y_{n-1},y_n,x_n) - log(Z))
# \end{align}

# Z is the sum of \begin{equation}exp( \Sigma^N_{n=1} \Sigma^F_{i=1} \lambda_i f(y_{n-1},y_n,x_n)\end{equation} for all possible sequence values: y1,y2,...yn. That is,
# 
# \begin{equation}
# Z = \Sigma^S_{s=1}exp( \Sigma^N_{n=1} \Sigma^F_{i=1} \lambda_i f(y_{s,n-1},y_{s,n},x_n)
# \end{equation}
# 
# Where s is the sequence no. ysn indicates the nth position of the sth sequence. 

# In place of 
# 
# \begin{equation}
# \Sigma^F_{i=1} \lambda_i f(y_{n-1},y_n,x_n)
# \end{equation}
# 
# I am using:
# 
# \begin{equation}
# l_1 GRUscores(x_n) + l_2 Transition_{y_{n-1},y_n}
# \end{equation}

# l1, and l2 are learnable parameters. Transition_yn-1,yn represents Transition_matrix[y_n-1][y_n] which constitutes a score for y_n (a given tag at position n) given that the previous tag (at n-1 position) is y_n-1.
# Transition_matrix is a matrix of learnable parameters.

# The first part of the CRF layer, computes log(Z) and the second part computes the unormalized probability score of the target sequence (if we are testing, validation or training).
# The second part is a straight forward summation:
# 
# \begin{equation}
# score = \Sigma^N_{n=1} l_1 GRUscores(x_n) + l_2 Transition_{y_{n-1},y_n}
# \end{equation}
# 
# This score is then used in the cost function as defined before. The cost function is basically:
# -(score - log(Z))

# The first part, that is, computing the Z of log(Z) is a bit troublesome. I used Dynamic programming is for this. To describe it briefly:
#     
# To calculate a the **sum of all sequences of size i,and ending with tag K** (S), I can use this:
# 
# ```
# for all j: 
# e = exp(l1*GRU_score(position_i,tag_k) + Transition[j][K])
# Sk += e*sum_of_all_sequences_of_size_(i-1)_ending_with_j
# ```   
# where sum_of_all_sequences_of_size_(i-1)_ending_with_j =
# ```
# exp(unexponentiated score of i-1 sized seq 1 ending with j) + exp(unexponentiated score of i-1 sized seq 2 ending with j).....
#     
# ```
# As a result of multiplying e with sum_of_all_sequences_of_size_(i-1)_ending_with_j we get:
# ```
# exp(l1*GRU_score(position_i,tag_k)+ Transition[j][K])*exp(unexponentiated score of i-1 sized seq 1 ending with j) + exp(l1*GRU_score(position_i,tag_k)+ Transition[j][K])*exp(unexponentiated score of i-1 sized seq 2 ending with j).....
# ```
# Which is:
# 
# ```
# exp(l1*GRU_score(position_i,tag_k)+ Transition[j][K] + unexp score of i-1 sized seq 1 ending with j) +
# exp(l1*GRU_score(position_i,tag_k)+ Transition[j][K] + unexp score of i-1 sized seq 2 ending with j).....
# ```
# Which is:
# ```
# exp(unexp score of i sized seq 1 ending with j) +
# exp(unexp score of i sized seq 2 ending with j).....
# ```
# 
# Now, we can do the same thing by adding logs:
# 
# ```
# for all j: 
# unexp_e = l1*GRU_score(position_i,tag_k) + Transition[j][K]
# Sk += exp(unexp_e+log(sum_of_all_sequences_of_size_(i-1)_ending_with_j))
# ```
# Because:
# ```
# exp(unexp_e+log(sum_of_all_sequences_of_size_(i-1)_ending_with_j))  
# = exp(unexp_e)*exp(log(sum_of_all_sequences_of_size_(i-1)_ending_with_j))  
# = exp(unexp_e)*(sum_of_all_sequences_of_size_(i-1)_ending_with_j)  
# Which is equal to e*(sum_of_all_sequences_of_size_(i-1)_ending_with_j) 
# ```
# Doing this can bring more stability. For more stability I did something more:
# ```
# let mj = unexp_e+log(sum_of_all_sequences_of_size_(i-1)_ending_with_j)
# So we get Sk = exp(m1) + exp(m2)....
# ```
# 
# exponentials of big no.s can can bring numerical instability, so I did this:
# 
# ```
# m_max = maximum(m1,m2....)
# ```
# Sk = exp(m1-m_max) + exp(m2-m_max).....
# While taking the log(Sk) to add it to other components while computing Sk values for next sequence,
# I use:
# 
# ```
# m_max + log(Sk)
# 
# ```
# 
# m_max + log(exp(m1-m_max) + exp(m2-m_max).....) is equivalent to log(exp(m1) + exp(m2).....) since:
# 
# ```
# m_max + log(exp(m1-m_max) + exp(m2-m_max).....)  
# = log(exp(m_max)) + log(exp(m1-m_max) + exp(m2-m_max).....)
# = log ( exp(m_max) * (exp(m1-m_max) + exp(m2-m_max).....))
# = log ( exp(m_max + m1 - m_max) + exp(m_max+m2-m_max)......)
# = log ( exp(m1) + exp(m2)....)
# ```
# 

# In[9]:



def fn1(tf_batch_size,seq_len,y_score,scores):
    
    batch_indices = tf.range(tf_batch_size)
        
    batch_indices = tf.reshape(batch_indices,[-1,1])
    tags_to_concat = tf.reshape(tf_tags[:,0],[-1,1])
        
    indices_batch_tag_i = tf.concat([batch_indices,tags_to_concat],1)
        
    y_score = (tf.gather_nd(scores[0],indices_batch_tag_i))
    y_score = l1*y_score
        
    i=tf.constant(1)
        
    def cond(i,y_score):
                
        return i<seq_len
            
    def body(i,y_score):
            
        batch_indices = tf.range(tf_batch_size)
        batch_indices = tf.reshape(batch_indices,[-1,1])
        tags_to_concat = tf.reshape(tf_tags[:,i],[-1,1])
            
        indices_batch_tag_i = tf.concat([batch_indices,tags_to_concat],1)
            
        tags_i_1 = tf.reshape(tf_tags[:,i-1],[-1,1])
        tags_i = tf.reshape(tf_tags[:,i],[-1,1])
            
        indices_i_1_to_i = tf.concat([tags_i_1,tags_i],1)
        
        GRU_scores = tf.gather_nd(scores[i],indices_batch_tag_i)
        T_scores = tf.gather_nd(Transition_matrix,indices_i_1_to_i)
        
        y_score +=  l1*GRU_scores + l2*T_scores
            
        return i+1, y_score
        
    _,y_score = tf.while_loop(cond,body,[i,y_score])
        
    return y_score
    
def fn2(y_score):
    return y_score


def model(traintestval):

    # GRU LAYER
    
    tf_batch_size = tf.shape(tf_sentences)[0]
    seq_len = tf.shape(tf_sentences)[1]
    
    hidden = tf.zeros([tf_batch_size,hidden_size],tf.float32) 
    
    hidden_list = bi_GRU(tf_sentences,hidden,seq_len,'Bi_GRU')
    
    hidden_list = tf.reshape(hidden_list,[-1,2*hidden_size])
    
    scores = tf.nn.relu(tf.matmul(hidden_list,W_score) + B_score)
    
    GRU_scores = tf.reshape(scores,[tf_batch_size,seq_len,len(NER_tags)])
    
    GRU_scores = layer_norm(GRU_scores,"GRU_out")
    
    # CRF LAYER
    
    scores =tf.transpose(GRU_scores,[1,0,2])

    
    # now scores shape = seq_len x batch_size x ner_tags
    
    scores = scores*scale_down
    
    i_1_scores = l1*scores[0]
    
    # now i_1_scores = batch_size x ner_tags
    
    i = tf.constant(1)
    
    def cond(i,i_1_scores):
        return i < seq_len
    
    def body(i,i_1_scores):
        
        # j -> k 
        
        i_scores = []
        
        for k in xrange(len(NER_tags)):
            
            score_k = []
            
            for j in xrange(len(NER_tags)):
                
                score_k.append(l1*scores[i,:,k] + l2*Transition_matrix[j,k] + i_1_scores[:,j])
                
            score_k = tf.convert_to_tensor(score_k)
            
            score_k = tf.transpose(score_k)
            
            max_k = tf.reduce_max(score_k,1)
            
            score_k = score_k-tf.reshape(max_k,[-1,1])
            score_k = tf.exp(score_k)
            
            score_k = tf.reduce_sum(score_k,1)
            
            score_k = max_k + tf.log(score_k)
                
            i_scores.append(score_k)
        
        i_1_scores = tf.convert_to_tensor(i_scores)
        
        # now i_1_scores = NER_tags_len x batch_size
        
        i_1_scores = tf.transpose(i_1_scores)
        
        return i+1,i_1_scores
    
    _,seq_len_scores = tf.while_loop(cond,body,[i,i_1_scores])
    
    # now seq_lens = batch_size x NER_tags_len
    max_scores = tf.reduce_max(seq_len_scores,1)
    seq_len_scores = seq_len_scores - tf.reshape(max_scores,[-1,1])
    seq_len_scores = tf.exp(seq_len_scores)
    Z = tf.reduce_sum(seq_len_scores,1)
    logZ = max_scores + tf.log(Z)
        
    y_score = tf.zeros([tf_batch_size])

        
    y_score = tf.cond(tf.equal(traintestval,True),
                      lambda:fn1(tf_batch_size,seq_len,y_score,scores),
                      lambda:fn2(y_score))


    return GRU_scores,logZ,y_score

                      


# In[10]:


GRU_scores,logZ,y_score = model(traintestval)

log_PyX = y_score - logZ

# l2 regularization
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
regularization = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

# Define loss and optimizer

#GRU_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=GRU_scores, labels=tf_tags))+regularization
cost = -tf.reduce_mean(log_PyX) + regularization


#optimizer_GRU = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(GRU_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# ### Prediction function:
# 
# This function computes the most probable sequence given the GRU scores and the transition matrix values. 
# While iterating through each position the code, keeps the maximum scoring sequence ending with tag k at i_scores[k].
# At the end, we will have the maximum scoring sequences ending with tag K (where k=0:no.of_classes-1).
# We can then take the maximum out of it.
# The paths of the sequence ending at k, are stored in Paths[:,k] which is updated at every step. 
# 
# 

# In[11]:


def predict(T,GRU_scores):
    
    batch_size = GRU_scores.shape[0]
    seq_len = GRU_scores.shape[1]
    
    GRU_scores = np.transpose(GRU_scores,[1,0,2])
    
    GRU_scores = GRU_scores*scale_down
    
    i_1_scores = GRU_scores[0]
    
    paths = np.zeros((batch_size,len(NER_tags),seq_len),np.int32)
    
    for b in xrange(batch_size):
        
        for i in xrange(len(NER_tags)):
            
            paths[b,i,0] = i
    
    for i in xrange(len(GRU_scores)):
        
        new_paths = np.zeros((batch_size,len(NER_tags),seq_len),np.int32)
        i_1_scores_temp = np.copy(i_1_scores)
        
        for k in xrange(len(NER_tags)):
            
            score_k = []
            
            for j in xrange(len(NER_tags)):
                
                score_ijk = GRU_scores[i,:,k]+T[j,k]+i_1_scores[:,j]
                score_k.append(score_ijk)
                
            score_k = np.asarray(score_k,np.float32)
            # score_k size = NER_tags x batch_size
            
            score_k = np.transpose(score_k)
            
            best_j = np.argmax(score_k,1)
            
            k_max_scores = np.amax(score_k,1)
            
            for b in xrange(batch_size):
                
                new_paths[b,k] = paths[b,best_j[b]]
                new_paths[b,k,i] = k
                i_1_scores_temp[b,k] = k_max_scores[b]
            
        paths = np.copy(new_paths)
        i_1_scores = np.copy(i_1_scores_temp)
    
    optimal_seq = np.zeros((batch_size,seq_len),np.int32)
    
    # scores_i_1 = NER_tags x batch_size
    
    for b in xrange(batch_size):
        
        best_seq_end_tag = np.argmax(i_1_scores[b,:])
        optimal_seq[b] = paths[b,best_seq_end_tag]
        
    return optimal_seq   


# In[12]:


def measure_plain_acc(predicted_seq,tags):
    
    acc_tensor = np.equal(predicted_seq,tags)
    acc_tensor.astype(np.float32)
    acc = np.mean(acc_tensor)
    return acc


# ### Training......

# In[13]:


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 

    sess.run(init) #initialize all variables
    step = 1   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_loss=2**30
    prev_val_loss=2**30
    patience = 20
    impatience = 0
    display_step = 20
    epochs = 200
            
    batch_size = 100
    
    while step <= epochs:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0

        batches_sentences,batches_tags = create_batches(train_sentences,train_tags,batch_size)

        for i in xrange(len(batches_sentences)):
            
            # Run optimization operation (backpropagation)
            _,loss,GRU_scores_out,T,npZ,npy_score = sess.run([optimizer,cost,GRU_scores,
                                                          Transition_matrix,logZ,log_PyX],
                                      feed_dict={tf_sentences: batches_sentences[i], 
                                                  tf_tags: batches_tags[i],
                                                  traintestval: True})

            total_loss += loss
            
            predicted_seq = predict(T,GRU_scores_out)
            acc = measure_plain_acc(predicted_seq,batches_tags[i])
            
            total_acc += acc
                
            if i%display_step == 0:
                print "Iter "+str(i)+", Loss= "+                      "{:.3f}".format(loss)+", Accuracy= "+                      "{:.3f}".format(acc*100)
                        
        avg_loss = total_loss/len(batches_sentences) 
        avg_acc = total_acc/len(batches_sentences)  
        
        loss_list.append(avg_loss) 
        acc_list.append(avg_acc) 

        val_batch_size = 100
        batches_sentences,batches_tags = create_batches(val_sentences,val_tags,batch_size)
        
        for i in xrange(len(batches_sentences)):
            
            # Run optimization operation (backpropagation)
            val_loss,val_GRU_scores,val_T = sess.run([cost,GRU_scores,Transition_matrix],
                                       feed_dict={tf_sentences: batches_sentences[i], 
                                                  tf_tags: batches_tags[i],
                                                  traintestval: True})
            total_val_loss += val_loss
            
            predicted_seq = predict(val_T,val_GRU_scores)
            val_acc = measure_plain_acc(predicted_seq,batches_tags[i])
            
            total_val_acc += val_acc
                      
            
        avg_val_loss = total_val_loss/len(batches_sentences) 
        avg_val_acc = total_val_acc/len(batches_sentences) 
             
        val_loss_list.append(avg_val_loss) 
        val_acc_list.append(avg_val_acc) 
    

        print "\nEpoch " + str(step) + ", Validation Loss= " +                 "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " +                 "{:.3f}%".format(avg_val_acc*100)+""
        print "Epoch " + str(step) + ", Average Training Loss= " +               "{:.3f}".format(avg_loss) + ", Average Training Accuracy= " +               "{:.3f}%".format(avg_acc*100)+""
        
        impatience += 1
        
        if avg_val_loss <= best_val_loss: 
            impatience = 0
            best_val_loss = avg_val_loss
            saver.save(sess, 'CRF_Model_Backup/model.ckpt') 
            print "Checkpoint created!"  
        
        if impatience > patience:
            print "\nEarly Stopping since best validation loss not decreasing for "+str(patience)+" epochs."
            break
            
        print ""
        step += 1
        
    
        
    print "\nOptimization Finished!\n"
    
    print "Best Validation Loss: %.3f"%((best_val_loss))
    


# ### Testing.....
# 
# Note: This is just the normal accuracy. I haven't calculated the F1 score which will probably be much lower.
# The dataset is pretty skewed. First most words are not named entities and have the tag 'O' - that's natural, but also makes the data biased.
# ON TOP OF THAT, I used PADDING WITHOUT BUCKETING. 
# One can achieve a fairly high accuracy just by predicting all words as 'O'. F1 accuracy is needed to make a fair estimate.
# 
# I will implement it later.

# In[14]:


with tf.Session() as sess: # Begin session
    
    print 'Loading pre-trained weights for the model...'
    saver = tf.train.Saver()
    saver.restore(sess, 'CRF_Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    print '\nRESTORATION COMPLETE\n'
    
    print 'Testing Model Performance...'
    
    total_test_loss = 0
    total_test_acc = 0
    
    test_batch_size = 100 #(should be able to divide total no. of test samples without remainder)
    batches_sentences,batches_tags = create_batches(test_sentences,test_tags,test_batch_size)
        
    for i in xrange(len(batches_sentences)):
            
            # Run optimization operation (backpropagation)
            test_loss,T,GRU_out = sess.run([cost,Transition_matrix,GRU_scores],
                                       feed_dict={tf_sentences: batches_sentences[i], 
                                                  tf_tags: batches_tags[i],
                                                  traintestval: True})
            total_test_loss += test_loss
            predicted_seq = predict(T,GRU_out)
            test_acc = measure_plain_acc(predicted_seq,batches_tags[i])
            total_test_acc += test_acc
                      
            
    avg_test_loss = total_test_loss/len(batches_sentences) 
    avg_test_acc = total_test_acc/len(batches_sentences) 


    print "\nTest Loss= " +           "{:.3f}".format(avg_test_loss) + ", Test Accuracy= " +           "{:.3f}%".format(avg_test_acc*100)+""


# ### Prediction on a random test sample
# 
# The model seems to be working at least.
# This cell can be run more than one times, and the prediction of different test sequences can be compared with the actual tags.
# 

# In[18]:


import random

rand = random.randint(0,len(test_sentences))

print "SENTENCE ABOUT TO BE FED:\n"
print map(vec2word,test_sentences[rand])
    
print "\n"

with tf.Session() as sess: # Begin session

    saver = tf.train.Saver()
    saver.restore(sess, 'CRF_Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    scores,T = sess.run([GRU_scores,Transition_matrix],
                          feed_dict={tf_sentences: [test_sentences[rand]],
                          tf_tags: [test_tags[rand]],
                          traintestval: True})
    prediction = predict(T,scores)
    acc = measure_plain_acc(prediction,[test_tags[rand]])

print "\nRESULT:\n"

for i in xrange(len(prediction[0])):
    ##removing Prediction of pads
    if vocab.index(vec2word(test_sentences[rand,i])) == vocab.index('<PAD>'):
        break
    print vec2word(test_sentences[rand,i]),
    print "- Tag: "+NER_tags[test_tags[rand,i]],
    print "Prediction: "+NER_tags[prediction[0,i]]
    
    
print "\n\nACCURACY: "+str(acc*100)+"%"


# ### Prediction
# 
# A very lame implementation of a cell that predicts user given sentences.
# One has to enter a sentence in the code in a tokenized lower case format.
# I may polish this part later, and make it more user friendly, or not.

# In[16]:


import random

rand = random.randint(0,len(sentences))

sentence = ['the','british','is','planning','to','attack','india','at','eleven']
# ENTER YOUR OWN SENTENCE HERE IN THIS FORMAT; LOWER CASE PLEASE.

print "SENTENCE ABOUT TO BE FED: "
print sentence
    
print "\n"

sentence_vec = map(word2vec,sentence)

with tf.Session() as sess: # Begin session

    saver = tf.train.Saver()
    saver.restore(sess, 'CRF_Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    T,scores = sess.run([Transition_matrix,GRU_scores],
                          feed_dict={tf_sentences: [sentence_vec],
                                     traintestval: False})
    prediction = predict(T,scores)

print "\nRESULT:\n"
for i in xrange(len(prediction[0])):
    print sentence[i],
    print "- "+NER_tags[prediction[0,i]]
    

