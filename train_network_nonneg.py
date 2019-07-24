# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:51:19 2019

@author: cklas
"""

#import class_nn_T_prediction_13_Delta_simple as nnp
#nnpredict=nnp.NNPredict()
##nnpredict.predict([1]*2,[1]*2)
#nnpredict.train_model(1000)


#import class_nn_T_prediction_monoton_02_5xn as nnp
nnpredict=NNPredict()
#nnpredict=nnp.NNPredict()
nnpredict.loaddatafromfile()
nnpredict.train_model(10)