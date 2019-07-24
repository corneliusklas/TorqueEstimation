rz6# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:51:19 2019

@author: cklas
"""

#import class_nn_T_prediction_15_Delta_simple as nnp
import class_nn_T_prediction_monoton as nnp
nnpredict=nnp.NNPredict()
#nnpredict.predict([1]*2,[1]*2)
#nnpredict.train_model(10)
nnpredict.plot_model()
for n in range(-5,5,1):nnpredict.plot_slice(n) #plot slices at different speeds