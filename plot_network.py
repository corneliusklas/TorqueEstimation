# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:51:19 2019

@author: cklas
"""


import class_nn_T_prediction_nonneg_2_hard_sigmoid as nnp
nnpredict=nnp.NNPredict()
#nnpredict.predict([1]*2,[1]*2)
#nnpredict.train_model(10)

nnpredict.loaddatafromfile() #load form "datasave.pickle"
#otherwise the data is loaded from the files specified in "class_nn_T_prediction_nonneg_2_hard_sigmoid.py", that takes a little bit of time

nnpredict.plot_model()
for n in range(-5,5,1):nnpredict.plot_slice(n) #plot slices at different speeds