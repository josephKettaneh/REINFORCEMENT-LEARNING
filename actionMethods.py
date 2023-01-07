#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def action_method(eps, Q, N,R, C, k_times, n_arms ,action):
    if(action == "optimistic" or action == "epsilon_greedy" ):
        if(action == "optimistic"):
            eps = 0
        
        for i in range(k_times):

            prob = np.random.rand()

            if prob < (1 - eps):
                a = np.argmax(Q)
                N[a]+=1
                Q[a]+=(1/N[a])*(R[i,a]-Q[a])
            else:

                a = np.random.randint(n_arms)
                N[a]+=1
                Q[a]+=(1/N[a])*(R[i,a]-Q[a])
                
    else:
        for i in range(k_times) :
            a = np.argmax(Q +C*np.sqrt(np.log(k_times)/N))
            A = R[i,a]
            N[a]+=1
            Q[a]+= 1/N[a]*(A-Q[a])
        
    return Q, N


# In[ ]:




