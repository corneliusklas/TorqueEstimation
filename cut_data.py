# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:37:59 2019

@author: cklas
"""

border=4

nnpredict=NNPredict()

nnpredict.loaddatafromfile()
data=nnpredict.data
stay=[]

datacut=[[],[],[],[],[],[]]
new=[[],[],[],[],[],[]]
i=0
datacut[i]= data[i].tolist()
i=1
datacut[i]= data[i].tolist()
i=2
datacut[i]= data[i].tolist()
i=3
datacut[i]= data[i].tolist()
i=4
datacut[i]= data[i].tolist()
i=5
datacut[i]= data[i].tolist()

#data[1]: label

#cut test data
while i< (len(datacut[1])-1):
    i+=1
    if abs(datacut[1][i])< border:
        stay.append(i)
        
new[0] = np.array([datacut[0][index] for index in stay])
new[1] = np.array([datacut[1][index] for index in stay])
new[4] =np.array( [datacut[4][index] for index in stay])

#cut train data
while i< (len(datacut[3])-1):
    i+=1
    if abs(datacut[3][i])< border:
        stay.append(i)
        
new[3] = np.array([datacut[3][index] for index in stay])
new[2] = np.array([datacut[2][index] for index in stay])
new[5] =np.array( [datacut[5][index] for index in stay])
        

               
nnpredict.data=new
nnpredict.savedatatofile()