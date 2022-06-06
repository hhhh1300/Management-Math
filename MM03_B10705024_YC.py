#!/usr/bin/env python
# coding: utf-8

# ##Q1
# 
# ###(a)
# 
# ###What you think 
# Generating six sentences,  
# • A = “I love Management Mathematics”  
# • B = “I hate Management Mathematics”  
# • C = “I love Operations Research”  
# • D = “I hate Operations Research”  
# • E = “I love Manufacturing Data Science”  
# • F = “I hate Manufacturing Data Science”  
# 
# ###Code

# In[5]:


import numpy as np
A = "I love Management Mathematics"
B = "I hate Management Mathematics" 
C = "I love Operations Research"  
D = "I hate Operations Research"  
E = "I love Manufacturing Data Science"  
F = "I hate Manufacturing Data Science"  


# ###Explain your answer 
# We generate the six sentences.

# ###(b)
# 
# ###What you think 
# By ```split()``` and ```lower()```, get a list of the words. 
# 
# ###Code

# In[6]:


wordsA = A.lower().split()
wordsB = B.lower().split()
wordsC = C.lower().split()
wordsD = D.lower().split()
wordsE = E.lower().split()
wordsF = F.lower().split()
print("wordsA", wordsA, sep=' : ')
print("wordsB", wordsB, sep=' : ')
print("wordsC", wordsC, sep=' : ')
print("wordsD", wordsD, sep=' : ')
print("wordsE", wordsE, sep=' : ')
print("wordsF", wordsF, sep=' : ')


# ###Explain your answer 
# We successfully got a list of words

# ###(C)
# 
# ###What you think 
# Identify the dimensionality of the feature space by the set class
# 
# ###Code

# In[7]:


vocab = set(wordsA)
vocab = vocab.union(set(wordsB), set(wordsC), set(wordsD), set(wordsE), set(wordsF))
print("vocab", vocab, sep=' : ')


# ###Explain your answer 
# We successfully print all vocabulary

# ###(D)
# 
# ###What you think 
# Use the ```list()``` in order to build the one-to-one mapping between ids and features.
# 
# ###Code

# In[8]:


vocab = list(vocab)
print(vocab)


# ###Explain your answer 
# We successfully convert the vocabulary into unique features

# ###(E)
# 
# ###What you think 
# Declare six 1D-arrays with increasing the corresponding feature value.
# 
# ###Code

# In[9]:


vA = np.zeros(len(vocab), dtype=float)
vB = np.zeros(len(vocab), dtype=float)
vC = np.zeros(len(vocab), dtype=float)
vD = np.zeros(len(vocab), dtype=float)
vE = np.zeros(len(vocab), dtype=float)
vF = np.zeros(len(vocab), dtype=float)
for w in wordsA:
    i = vocab.index(w)
    vA[i] += 1
for w in wordsB:
    i = vocab.index(w)
    vB[i] += 1
for w in wordsC:
    i = vocab.index(w)
    vC[i] += 1
for w in wordsD:
    i = vocab.index(w)
    vD[i] += 1
for w in wordsE:
    i = vocab.index(w)
    vE[i] += 1
for w in wordsF:
    i = vocab.index(w)
    vF[i] += 1
print("vA", vA, sep=' : ')
print("vB", vB, sep=' : ')
print("vC", vC, sep=' : ')
print("vD", vD, sep=' : ')
print("vE", vE, sep=' : ')
print("vF", vF, sep=' : ')


# ###Explain your answer 
# We successfully build each word vector

# ###(F)
# 
# ###What you think 
# Compute the cosine similarity
# 
# ###Code

# In[10]:


def cosSimi(a, b) :
    return np.dot(a, b)/np.sqrt(np.dot(a, b)/np.sqrt(np.dot(a, b)))
cosA = [1, cosSimi(vA, vB), cosSimi(vA, vC), cosSimi(vA, vD), cosSimi(vA, vE), cosSimi(vA, vF)]
cosB = [cosSimi(vA, vB), 1, cosSimi(vB, vC), cosSimi(vB, vD), cosSimi(vB, vE), cosSimi(vB, vF)]
cosC = [cosSimi(vA, vC), cosSimi(vB, vC), 1, cosSimi(vC, vE), cosSimi(vC, vE), cosSimi(vC, vF)]
cosD = [cosSimi(vA, vD), cosSimi(vB, vD), cosSimi(vC, vD), 1, cosSimi(vD, vE), cosSimi(vD, vF)]
cosE = [cosSimi(vA, vE), cosSimi(vB, vE), cosSimi(vC, vE), cosSimi(vD, vE), 1, cosSimi(vE, vF)]
cosF = [cosSimi(vA, vF), cosSimi(vB, vF), cosSimi(vC, vF), cosSimi(vD, vF), cosSimi(vE, vF), 1]
import pandas as pd
data = np.vstack((np.mat(cosA), np.mat(cosB), np.mat(cosC), np.mat(cosD), np.mat(cosE), np.mat(cosF)))
df = pd.DataFrame(data.T , index = ["A", "B", "C", "D", "E", "F"], columns = ["A", "B", "C", "D", "E", "F"] )
print(df)


# ###Explain your answer 
# We successfully compute the cosine similarity of each pair

# ###(G)
# 
# ###What you think 
# Answer what I obseverd
# 

# ###Explain your answer 
# Found the relationship between six sentences

# ##Q2
# 
# ###(a)
# 
# ###What you think 
# Build the frequency of key words matrix and convert all the column vectors into unit vectors 
# 
# ###Code

# In[48]:


def buildVector(v, arr) :
    v = np.array(arr)
    v = (v / np.linalg.norm(v))
    return v
v1, v2, v3, v4, v5, v6, v7, v8 = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
v1 = buildVector(v1, [0,0,5,6,0,0,0,5,0,0])
v2 = buildVector(v2, [6,0,4,5,0,0,0,3,0,4])
v3 = buildVector(v3, [3,0,4,3,0,0,5,3,0,4])
v4 = buildVector(v4, [0,0,5,3,0,0,2,2,5,3])
v5 = buildVector(v5, [1,0,4,4,3,4,3,4,1,4])
v6 = buildVector(v6, [0,5,0,4,0,6,3,2,3,1])
v7 = buildVector(v7, [1,3,3,3,4,0,0,1,1,0])
v8 = buildVector(v8, [1,2,3,2,3,2,1,1,0,3])
Q = np.vstack((np.mat(v1), np.mat(v2), np.mat(v3), np.mat(v4), np.mat(v5), np.mat(v6), np.mat(v7), np.mat(v8))).T
print("The frequency of key words matrix Q", Q, sep=':\n')


# ###Explain your answer 
# Successfully build the matrix  

# ###(b)
# 
# ###What you think 
# Form a unit vector that searchs the keywords determinants, matrices, and systems.
# 
# ###Code

# In[58]:


x = (np.array([[1],[0],[0],[1],[0],[0],[0],[1],[0],[0]]))
print("Unit search vector x ", x, sep=':\n')


# ###Explain your answer 
# Successfully build the unit vector

# ###(c)
# 
# ###What you think 
# Calculate the cosine similarity and rank
# 
# ###Code

# In[61]:


print("The cosine similarity :\n", Q.T * x)
print("Rank of module : ", np.linalg.matrix_rank(Q))


# ###Explain your answer 
# Successfully calculate the cosine similairy and rank of module
