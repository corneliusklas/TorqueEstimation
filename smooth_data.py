# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:37:59 2019

@author: cklas
"""



nnpredict=NNPredict()

nnpredict.loaddatafromfile()
data=nnpredict.data

for i in range(3):
    
    for i in range(1,len(data[0])-1):
        #smooth test data
        data[0][i] =np.mean([data[0][i-1],data[0][i],data[0][i+1]],axis=0) 
        #smooth train data
        data[2][i] =np.mean([data[2][i-1],data[2][i],data[2][i+1]],axis=0) 
        #np.mean([np.array([1,2,3]),np.array([2,3,4])],axis=0)
        
               
nnpredict.data=data
nnpredict.plot_model()
#nnpredict.savedatatofile()