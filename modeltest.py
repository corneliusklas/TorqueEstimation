# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:18:35 2019

@author: cklas
"""

#nnpredict.model_T.predict(np.array([[n, pwm, n, pwm, pwmxn]])).flatten()[0]


nnpredict.model_T.predict(np.array([[0.3, 1, 0.3, 1, 0]])).flatten()[0]
#Out[10]: 10.802639  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[3]: 10.948788   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-1, 0.5, -1, 0.5, 0]])).flatten()[0]
#Out[13]: -5.847395  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[7]: -5.794898   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[1, 0.5, 1, 0.5, 0]])).flatten()[0]
#Out[14]: 11.645586  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[8]: 11.813379   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[0.1, 0.2, 0.1, 0.2, 0]])).flatten()[0]
#Out[17]: 6.157156  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[10]: 6.160616   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[0.1, 0.02, 0.1, 0.02, 0]])).flatten()[0]
#Out[18]: 3.0114336  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[11]: 3.0017996   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0.1, -0.02, -0.1, -0.02, 0]])).flatten()[0]
#Out[19]: -3.3863525  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[12]: -3.3617249   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0, -0, -0, -0, 0]])).flatten()[0]
#Out[23]: 0.23437405  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[13]: 0.19550514   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0, -0, -0, -0, -1]])).flatten()[0]
#Out[24]: -0.4028101  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[14]: -0.43778133   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0, -0, -0, -0, 1]])).flatten()[0]
#Out[25]: 0.8407078  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[15]: 0.8169508   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0, 0.5, -0, 0.5, 1]])).flatten()[0]
#Out[28]: 5.883444  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[16]: 5.847205   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5
nnpredict.model_T.predict(np.array([[-0, -0.5, -0, -0.5, -1]])).flatten()[0]
#Out[29]: -5.9107695  #NN_T_predict_nonneg_hard_sigmoid_16.h5
#Out[17]: -5.9338393   #NN_T_predict_nonneg_hard_sigmoid_16_mae012.h5



