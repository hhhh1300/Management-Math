#!/usr/bin/env python
# coding: utf-8

# ##(a)
# ###What you think 
# Create three random variable with normal distribution, which are  
# (1)high school math GPA,  
# (2)binary indicator variable for whether they took calculus in high school or not,  
# (3)binary indicator variable for whether they took NTU’s precalculus class or not, respectively.  
# Then determine the dependent variable "NTU Calculus I GPA grade," which follow the eqution 
# $NTU_Calculus_I_grade𝑖=0.3+0.7⋅HS_math_GPA𝑖+0.3⋅HS_calculus𝑖+0.1⋅NTU_precalculus𝑖+𝜀𝑖$,   
# where $𝜀𝑖∼Normal(mean=0, Standard Deviation=0.5)$
# 
# ###Code

# In[65]:


import numpy as np
size = 2000
HS_math_GPA = np.random.normal(3.5, 0.5, size)
HS_calculus = np.zeros(size)
NTU_precalculus = np.zeros(size)
NTU_Calculus_I_grade = np.zeros(size)
true_sigma = 0.5
coefficient = {"Intercept":0.3,
               "hs_math_gpa":0.7,
               "hs_calculus":0.3,
               "uw_precalc":0.1}
for i in range(0, size) :
    if(HS_math_GPA[i] < 2) :
        HS_math_GPA[i] = 2
    elif(HS_math_GPA[i] > 4) :
        HS_math_GPA[i] = 4
        
    p = 0
    if(HS_math_GPA[i] > 3.6) :
        p = 0.75
    else :
        p = 0.4
    HS_calculus[i] = np.random.binomial(1, p, 1)
    
    if(HS_calculus[i] == 0) :
        p = 0.7
    elif(HS_math_GPA[i] < 3.5) :
        p = 0.6
    else :
        p = 0.25
    NTU_precalculus[i] = np.random.binomial(1, p, 1) #sigma follow N(0, 0,5)
    
    sigma = np.random.normal(0, true_sigma, 1)
    NTU_Calculus_I_grade[i] = coefficient["Intercept"]+coefficient["hs_math_gpa"]*HS_math_GPA[i]+coefficient["hs_calculus"]*HS_calculus[i]+coefficient["uw_precalc"]*NTU_precalculus[i]+sigma
    NTU_Calculus_I_grade[i] = np.round(NTU_Calculus_I_grade[i], 1)
    if(NTU_Calculus_I_grade[i] < 0) :
        NTU_Calculus_I_grade[i] = 0
    elif(NTU_Calculus_I_grade[i] > 4) :
        NTU_Calculus_I_grade[i] = 4
    elif(NTU_Calculus_I_grade[i] < 0.7) :
        NTU_Calculus_I_grade[i] = 0.7

import pandas as pd
data = np.vstack((NTU_Calculus_I_grade, HS_math_GPA, HS_calculus, NTU_precalculus))
df = pd.DataFrame(data.T , columns = ["NTU_Calculus_I_grade", "HS_math_GPA", "HS_calculus", "NTU_precalculus"] )
print(df)


# ###Explain your answer 
# I build three random variable with matirx and np.random, then determine the NTU_Calculus_I_grade.

# ##(b) 
# ###What you think 
# We can rewrite the line equation $y = mx + c$ as $y = AP$, where $A = [x$ $1]$ and $P = [[m], [c]]$ 
# Then by ```np.linalg.lstsq```, we can solve the $m$ and $c$     
# After the individual linear regression, then we can build the Multidimensional linear models.
# ###Code

# In[66]:


import matplotlib.pyplot as plt
A = np.vstack([HS_math_GPA, np.ones(size)]).T
m, c = np.linalg.lstsq(A, NTU_Calculus_I_grade, rcond=None)[0]
print("y = mx + c, m:", m, "c:", c)
plt.plot(HS_math_GPA, NTU_Calculus_I_grade, 'o', label='Original data', markersize=3)
plt.plot(HS_math_GPA, m*HS_math_GPA + c, 'r', label='Fitted line')
plt.xlabel('HS_math_GPA',fontsize=10)
plt.ylabel('NTU_Calculus_I_grade',fontsize=10)
plt.legend()
plt.show()

A = np.vstack([HS_calculus, np.ones(size)]).T
m, c = np.linalg.lstsq(A, NTU_Calculus_I_grade, rcond=None)[0]
print("y = mx + c, m:", m, "c:", c)
plt.plot(HS_calculus, NTU_Calculus_I_grade, 'o', label='Original data', markersize=3)
plt.plot(HS_calculus, m*HS_calculus + c, 'r', label='Fitted line')
plt.xlabel('HS_calculus',fontsize=10)
plt.ylabel('NTU_Calculus_I_grade',fontsize=10)
plt.legend()
plt.show()

A = np.vstack([NTU_precalculus, np.ones(size)]).T
m, c = np.linalg.lstsq(A, NTU_Calculus_I_grade, rcond=None)[0]
print("y = mx + c, m:", m, "c:", c)
plt.plot(NTU_precalculus, NTU_Calculus_I_grade, 'o', label='Original data', markersize=3)
plt.plot(NTU_precalculus, m*NTU_precalculus + c, 'r', label='Fitted line')
plt.xlabel('NTU_precalculus',fontsize=10)
plt.ylabel('NTU_Calculus_I_grade',fontsize=10)
plt.legend()
plt.show()

print("Multidimensional linear models(by numpy.linalg.lstsq) :")
X = np.mat([np.ones(size), HS_math_GPA, HS_calculus, NTU_precalculus]).T
Y = np.mat([NTU_Calculus_I_grade]).T
W = np.linalg.lstsq(X, Y, -1)[0]
print("y = β1 + β2 * HS_math_GPA + β3 * HS_calculus + β4 * NTU_precalculus")
print("β1:", W[0, 0], "β2:", W[1, 0], "β3:", W[2, 0], "β4:", W[3, 0])
print("residual_square by numpy.linalg.lstsq:", np.linalg.lstsq(X, Y, -1)[1])
print()
import statsmodels.api as sm
mod = sm.OLS(NTU_Calculus_I_grade, X)
res = mod.fit()
print("Multidimensional linear models(by statsmodels.api) :")
print(res.summary())


# ###Explain your answer 
# I solve the $m$ and $c$ by ```np.linalg.lstsq```, then visualize my answer by ```matplotlib.pyplot```.   
# Then solve the Multidimensional linear models with ```np.linalg.lstsq``` and ```statsmodels.api```.

# ##(c)
# ###What you think  
# Build the matrix $Χ$ (intercept plus independent variables) and variable $y$ (dependent variable)
# 
# ###Code

# In[67]:


X = np.mat([np.ones(size), HS_math_GPA, HS_calculus, NTU_precalculus]).T
y = np.mat([NTU_Calculus_I_grade]).T


# ###Explain your answer 
# Literally setup the matrix $X$ and $y$.

# ##(d)
# ###What you think  
# Compute matrix quantities $(𝑋^{T}𝑋)^{-1}$ by ```inv()``` and find $adj((𝑋^{T}𝑋)^{-1})$, $det((𝑋^{T}𝑋)^{-1})$, inverse of $(𝑋^{T}𝑋)^{-1}$
# 
# ###Code

# In[68]:


C = X.T * X
inv_C = np.linalg.inv(C)
det_C = np.linalg.det(C)
adj_C = det_C * inv_C
print("inv_C:")
print(inv_C)
print("det_C:")
print(det_C)
print("adj_C:")
print(adj_C)


# ###Explain your answer 
# Successfully find $adj((𝑋^{T}𝑋)^{-1})$, $det((𝑋^{T}𝑋)^{-1})$, inverse of $(𝑋^{T}𝑋)^{-1}$

# ##(e)
# ###What you think  
# Find $\hat{𝛽̂}$ = $(𝑋^{T}𝑋)^{-1}𝑋^{𝑇}𝑦$, residuals, residual variance $𝜎̂^2$,standard errors of the covariance matrix $Var\hat{𝛽̂}$.
# 
# ###Code

# In[69]:


import math
beta_hat = inv_C * X.T * y
p = 3
residual = 0
residual_square = 0
residual_var = 0
cov = 0 
N = X * beta_hat
for i in range(0, size) :
    residual = residual + Y[i] - N[i]
    residual_square = residual_square + (Y[i] - N[i])**2
print("residual_square:")
print(residual_square)
print("residual:")
print(residual[0, 0])
residual_var = (Y - N).T * (Y - N) / (size-p-1)
print("residual_var:")
print(residual_var[0, 0])
cov = residual_var[0, 0] * inv_C
print("Covariance:")
print(cov)
beta_SE = np.zeros(p+1)
for i in range(0, p+1) :
    beta_SE[i] = math.sqrt(cov[i, i])
print("beta_SE:")
print(beta_SE)


# ###Explain your answer 
# Successfully find $\hat{𝛽̂}$ = $(𝑋^{T}𝑋)^{-1}𝑋^{𝑇}𝑦$, residuals, residual variance $𝜎̂^2$,standard errors of the covariance matrix $Var\hat{𝛽̂}$.

# ##(f)
# ###What you think  
# Build a table of the true coefficient, the manual coefficient, and the coefficient computed by ```statsmodels.api``` to see what is different between them.
# 
# ###Code

# In[70]:


import pandas as pd
data = np.vstack(([0.3, 0.7, 0.3, 0.1], beta_hat.T, res.params))
df = pd.DataFrame(data.T ,index = ["intercept", "HS_math_GPA", "HS_calculus", "NTU_precalculus"], columns = ["Truth", "Manual", "lstsq"] )
print(df)


# ###Explain your answer 
# Observing the table, I find the coefficient we computed is the same as the coefficient computed by ```statsmodels.api```, and the coefficient we got is quite close to the true coefficient

# ##(g)
# ###What you think  
# Compute the results of the estimated standard errors $(\hat{Var}(\hat{𝛽̂}))^{0.5}$ and observe the result.
# 
# ###Code

# In[71]:


true_cov = true_sigma**2 * inv_C
true_SE = np.zeros(p+1)
for i in range(0, p+1) :
    true_SE[i] = math.sqrt(true_cov[i, i])
data = np.vstack((true_SE, beta_SE, res.bse)).T
df = pd.DataFrame(data ,index = ["intercept", "HS_math_GPA", "HS_calculus", "NTU_precalculus"], columns = ["Truth", "Manual", "lstsq"] )
print(df)


# ###Explain your answer 
# With ```statsmodels.api```, I find the standard error is the same as the standard error computed by ourselves, and the value is quite close to the true standard error.

# ##(f)
# ###What you think  
# Compute the results of the estimated standard errors $(\hat{Var}(\hat{𝛽̂}))^{0.5}$ and observe the result.
# 
# ###Code

# In[72]:


data = np.vstack((true_sigma**2, residual_var[0, 0], res.mse_resid)).T
df = pd.DataFrame(data, index = [""], columns = ["Truth", "Manual", "lstsq"] )
print(df)


# ###Explain your answer 
# With ```statsmodels.api```, I find the residual variance is the same as the residual variance computed by ourselves, and the value is quite close to the true residual variance.

# ##(g)
# ###What you think  
# Predict a new data and observe the result
# 
# ###Code

# In[73]:


HS_math_GPA_new = 3.123456
HS_calculus_new = 1
NTU_precalculus_new = 0
M = [1, HS_math_GPA_new, HS_calculus_new, NTU_precalculus_new]
NTU_Calculus_I_grade_new_manual = M * beta_hat
sigma = np.random.normal(0, true_sigma, 1)
NTU_Calculus_I_grade_new_true = coefficient["Intercept"]+coefficient["hs_math_gpa"]*HS_math_GPA_new+coefficient["hs_calculus"]*HS_calculus_new+coefficient["uw_precalc"]*NTU_precalculus_new+sigma
if(NTU_Calculus_I_grade_new_true < 0) :
    NTU_Calculus_I_grade_new_true = 0
elif(NTU_Calculus_I_grade_new_true > 4) :
    NTU_Calculus_I_grade_new_true = 4
elif(NTU_Calculus_I_grade_new_true < 0.7) :
    NTU_Calculus_I_grade_new_true = 0.7
data = [(NTU_Calculus_I_grade_new_true[0], NTU_Calculus_I_grade_new_manual[0, 0], (M * np.mat(res.params).T)[0, 0])]
df = pd.DataFrame(data ,index = [""], columns = ["Truth", "Manual", "lstsq"] )
print(df)


# ###Explain your answer 
# The prediction done by me and ```statsmodels.api``` is the same.  
# However, the difference between the prediction and the true value must be a $𝜀$, which follows $N(0, 0.5)$
