#!/usr/bin/env python
# coding: utf-8

# ##Q1

# ###(a)
# 
# ###What you think 
# Define $A$ matrix and $b$ vector. 
# 
# ###Code

# In[1]:


#(a)
import numpy as np
use_me = ([-5,4,0,1],[-30,27,2,7],[5,2,0,2],[10,1,-2,1])
A = np.array(use_me);
b = np.array([-17,-102,-7,-6])


# ###Explain your answer 
# We define the matrix $A$ and vector $b$ for the linear system $Ax = b$

# ###(b)
# 
# ###What you think 
# solve $Ax = b$ with ```linalg.solve```.  
# 
# ###Code

# In[2]:


#(b)
print(np.linalg.solve(A,b))


# ###Explain your answer 
# We solve the linear system $Ax = b$ with numpy.linalg.

# ###(c)
# 
# ###What you think 
# solve $A^{-1}$ with ```linalg.inv```.  
# 
# ###Code

# In[3]:


#(c)
print(np.linalg.inv(A))


# ###Explain your answer 
# We find $A^{-1}$ with numpy.linalg.

# ###(d)
# 
# ###What you think 
# calculate $A^{-1}b$.
# 
# ###Code

# In[4]:


#(d)
print(np.dot(np.linalg.inv(A), b))


# ###Explain your answer 
# We calculate $A^{-1}b$ and find the result from (b) and (d) is the same.

# In[5]:


get_ipython().system('pip install opencv-python')


# In[6]:


from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from numpy import asarray
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# ##Q2
# 
# ###(a)
# 
# ###What you think
# use Matirx Addtion
# 
# ###Code

# In[7]:


img1 = cv2.imread('ta.jpg')
img2 = cv2.imread('bg.jpg')
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
img = cv2.add(img1,img2_resized)

plt.imshow(img)
#https://blog.akanelee.me/posts/205816-methods-of-making-pattern/ (source of bg.jpg)


# ###Explain your answer 
# use Matrix Addtion to add one image to another image

# ###(b)
# 
# ###What you think
# use Matirx Subtraction
# 
# ###Code

# In[8]:


img = cv2.subtract(img1, img2_resized)
plt.imshow(img)


# ###Explain your answer 
# use Matrix Subtraction to subtract one image to another image

# ###(c)
# 
# ###What you think
# rotate the image by clockwise 60 degree with double scaling
# 
# ###Code

# In[9]:


(height, width) = img1.shape[:2]
img1_resized = cv2.resize(img1, (int(width*2), int(height*2)), interpolation = cv2.INTER_CUBIC)
(rows, cols) = img1_resized.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), -60, 1)
img1_resized = cv2.warpAffine(img1_resized, M, (cols, rows))
plt.imshow(img1_resized)


# ###Explain your answer 
# successfully rotate the image by clockwise 60 degree with double scaling

# ###(d)
# 
# ###What you think
# use Canny Edge Detection
# 
# ###Code

# In[10]:


edges = cv2.Canny(img1, 100, 200)
plt.subplot(121),plt.imshow(edges, cmap = "gray")


# ###Explain your answer 
# detect the edge by Canny Edge Detection
