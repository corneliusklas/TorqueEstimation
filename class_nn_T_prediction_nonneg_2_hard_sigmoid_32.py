# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:04:57 2018
https://www.tensorflow.org/tutorials/keras/basic_regression
@author: cklas
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

class NNPredict:


     #print(tf.__version__

    load=True # load network data
    path='recordings/Motor22/'
    netpath='Networks/'
    
    #basename =  input('Enter file basename: ')    
    #read the data_
    n_old=0 # 0: no data from previous timestep
    n_mean=20 #min:1 = no smooth
    u_mean=1 #min:1 = no smooth
    #nr of values fo data preperation
    nr_n=nr_u=max(n_old,n_mean)+1
    #new_model=False  # if new set ==False the old  model is used, but new test data or training data can be loaded
    #EPOCHS = 1000
    zukunft=0 # zukünftige werte für die Drehmomentschätzung - verschiebung des torque in die zukunf
    
    driven=True # das produkt aus pwm und drehzahl wird als parameter verwendet
#    power_direction=True  # vorzeichen Drehmoment aus linearem modell
#    acceleration=False # die Beschleunigung wird als eingabeparameter verwendet
    BatchNormalization=False
    #scale values to a good range
    n_factor= 3.0 #recalculated from old average and standard deviation n_factor=459.0*(1000*60/4096)/60*np.pi*2/231
    pwm_factor=-1700.0
    acc_factor=0.005
    #T_factor=0.6
    
    #the nn model for torque estimation
    model_T=0        
    
    filenames=["low_torque.csv","test1.csv", "test2.csv","test3.csv","test4.csv","zero_torque.csv","zero_torque_01.csv","zero_torque_1.csv","zero_torque_05.csv","zero_torque_5.csv","zero_torque_20.csv","zero_vel.csv","jeff_2019-06-14_12-14-45.csv","jeff_2019-06-14_13-56-31.csv"]#,"Torquecontrol_2019-06-19_16-51-11_A.csv","Torquecontrol_2019-06-19_16-51-11_B.csv"]
    testnumbers=[4]
    #trainbasename=    testbasename='0_M1_sum_all_nzero_no_x10.csv'
    #trainbasename=testbasename='0_M1_sum_all_Nzero.csv'
    savefilename="NN_T_predict_nonneg_hard_sigmoid_32"    
    
    history_T=[]
    
    data=[[],[],[],[]]
    
    pwmXn_old=0
    
    def __init__(self,Data_=[[],[],[],[]]):
        #self.data=self.readData(self.path, self.trainbasename, self.testbasename)
        self.model_T = self.build_model()
        self.model_T.summary()
        if (self.load):
            self.model_T.load_weights("./"+self.netpath+self.savefilename+'.h5')
        self.data=Data_
    
    def linearModel_dcx22(self,n,pwm):
        T=0
#        #ticks/ms -> U/min : 1000*60/4096
#        #rad/s -> U/min : 
#        n_f=60/np.pi/2*231 #?
#        #übersetzung getriebe +Nm->mNm
#        T_f=1000/231
#        #Drehzahlkonstante   [min-1 V-1] 
#        motor_a	= 226-60
#        #Kennliniensteigung [min-1 mNm-1] 
#        motor_b=123
#        #Wirkungsgrad Getriebe *Motor	
#        Motor_eta= 0.8# max from datasheet:0.75*0.85
#        #verlust im pwm schaltkreis
#        U_lost=0
#        #Spannung bei pwm max
#        Umax=48#+U_lost
#
#        pwmmax=3000
#        pwm_zero=100
#        
#        n=n *n_f
#        U=max(0,(abs(pwm)-pwm_zero)/(pwmmax-pwm_zero)*(Umax-U_lost))*np.sign(pwm)
#
#       
#        #U(M,n)=(n-b*M)/a
#        T_motor=(U*motor_a-n)/-motor_b
#        T=T_motor*Motor_eta/T_f
            
        return T
    
    def inertia_dcx22(self,n,n_1,dt):
        I= 5.952E-07 #kg*m²
        i=1 #jetzt schon inklusive; vorher 231 #uebersetzung
        n_f=1#(1000)/4096*2*np.pi #ticks/ms -> rad/s
        acc=(n-n_1)/dt*n_f
        #print (acc)
        T=acc*I*i
         
        return T
    
    def prepareData(self,n,u,pwmXn=0):
        l_n_mean=self.n_mean
        l_u_mean=self.u_mean

        n_old=self.n_old
        #datum=[n[i],u[i], n[i-1], n[i-2], n[i-3],u[i-1], u[i-2], u[i-3]]
        #this function gehts the last entries of n and u
        n=np.array(n)/self.n_factor
        u=np.array(u)/self.pwm_factor
        n_mean=np.average(n[-l_n_mean:len(n)])
        #n_smooth_old=np.average(n[-l_n_smooth-1:-1])
        u_mean=np.average(u[-l_u_mean:len(u)])
        #first the last entries
        datum=[n_mean, u_mean, n[-1],u[-1] ]
#        #then acceneration, if wanted
#        if self.acceleration:
#            acc= n[-1]-n[-2]
#            acc=acc/self.acc_factor
#            datum.append(acc)
#            acc_smooth=n_smooth-n_smooth_old
#            acc_smooth=acc_smooth/self.acc_factor
#            datum.append(acc_smooth)	
        #then the other entries of speed and pwm    
        for j in range( -n_old-1,-1):
            datum.append(n[j])
        for j in range( -n_old-1,-1):
            datum.append(u[j])
    
        if self.driven:
            #pwm x n    : drive or driven
            if pwmXn==0:
                pwmXn= np.sign(u[-1]*n[-1])
                pwmXn=self.pwmXn_old
            else:
                self.pwmXn_old=pwmXn
            datum.append(pwmXn)    
        
#        if self.power_direction:
#            powerd=np.sign(self.linearModel_dcx22(n[-1]*self.n_factor,u[-1]*self.pwm_factor))
#            datum.append(powerd)
        return datum
    
        
    def readData(self):
        path=self.path
        filenames= self.filenames
        testnumbers=self.testnumbers
        

        zukunft=self.zukunft 
        #acceleration=self.acceleration
        #driven=self.driven
        
        trainAndTestData=[]
        trainAndTestLabels=[]
        trainAndTestLabels_org=[]
        
        paths=[]
        for file in filenames:
             paths.append(path+file)
             
        for filenumber,filepath in enumerate(paths):
            data_ = np.genfromtxt(filepath, delimiter=';', skip_header=1, skip_footer=1, names=['marker','iteration','t','Tz','postition','pwm','Ticks','n','n_smooth','n_target'], dtype=None, encoding='utf-8')
            print('Data ',filenumber,filepath)
            if filenumber in testnumbers:
                print("Part of Testset")
            else:
                print ("Part of Training set")
            
            #remove outliers
#            for i in range(len(data_['Tz'])):
#                if abs(data_['Tz'][i]-self.linearModel_dcx22(data_['n'][i],data_['pwm'][i]))>0.5 or abs(data_['Tz'][i])>1:
#                    data_['Tz'][i]=np.nan
                
                #mirror test set
                #data_['Tz']=-data_['Tz']
                #data_['n']=-data_['n']
                #data_['pwm']=-data_['pwm']
                        
            #cut data to good range
#            pwmmax=2999
#            nmax=10
#            for i in range(len(data_)):
#                if abs(data_['pwm'][i])>pwmmax:
#                    data_['pwm'][i]=np.nan
#                if abs(data_['n'][i])>nmax:
#                    data_['n'][i]=np.nan
#                # if a new dataset begins there sould be no acceleration calulation over the borders
#                if data_['name'][i]!=data_['name'][i-1]:
#                    data_['n'][i]=np.nan
#                    data_['n'][i-1]=np.nan
                #if abs(data_['n'][i])>0 and data_['pwm'][i]<0 :
                #    data_['n'][i]=np.nan
                      
                #data_['Tz'][i]=data_['Tz'][i]
            
            
            #seperate interesting data
            #if zukunft>0:
            #    trainOrTestLabels_org=data_['Tz'][self.nr_n-zukunft:-zukunft] 
            #else:
            #    trainOrTestLabels_org=data_['Tz'][self.nr_n:] 
            
            trainOrTestData=[]
            trainOrTestLabels=[]
            trainOrTestLabels_org=[]
            n=data_['n']#/460
            u=data_['pwm']#/1700
            T=data_['Tz']
            

            for i in range(self.nr_n,len(data_)): 

                datum= self.prepareData(n[i-self.nr_n:i+1],u[i-self.nr_u:i+1])
                trainOrTestData.append(datum)
                trainOrTestLabels.append(T[i-zukunft]-self.linearModel_dcx22(datum[0]*self.n_factor,datum[1]*self.pwm_factor))
                trainOrTestLabels_org.append(T[i-zukunft])
                
            #remove values out of sensor range
            trainOrTestLabels=list(trainOrTestLabels)
            trainOrTestData=list(trainOrTestData)

            
            
#            #check for nan values
#            #try:
#            for i in range(len(trainOrTestLabels)):
#                if np.isnan(trainOrTestLabels[i])==False and np.isnan(sum(trainOrTestData[i]))==False:
#                    trainOrTestLabels_cut.append(trainOrTestLabels[i])
#                    trainOrTestData_cut.append(trainOrTestData[i])
#                    trainOrTestLabels_org_cut.append(trainOrTestLabels_org[i])
#                    
#            trainOrTestData =   trainOrTestData_cut
#            trainOrTestLabels=trainOrTestLabels_cut 
#            trainOrTestLabels_org=trainOrTestLabels_org_cut
   
            trainAndTestData.append(trainOrTestData)
            trainAndTestLabels.append(trainOrTestLabels)
            trainAndTestLabels_org.append(trainOrTestLabels_org)
#todo replace            
#            for i in range(len(trainOrTestLabels)):
#                if np.isnan(trainOrTestLabels[i])==True or np.isnan(sum(trainOrTestData[i]))==True:
#                    del trainOrTestLabels[i]
#                    del trainOrTestData[i]
#                    del trainOrTestLabels_org[i]
            
    
        test_data_=[]
        test_labels_=[]
        train_data_=[]
        train_labels_=[]
        test_labels_org_=[]
        train_labels_org_=[]
        
        for i in range(len(trainAndTestData)):
            if i in testnumbers:
                test_data_=test_data_+trainAndTestData[i]
                test_labels_=test_labels_+trainAndTestLabels[i]
                test_labels_org_=test_labels_org_+trainAndTestLabels_org[i]
                print("Part of Testset")
            else:
                train_data_=train_data_+trainAndTestData[i]
                train_labels_=train_labels_+trainAndTestLabels[i]
                train_labels_org_=train_labels_org_+trainAndTestLabels_org[i]
                print("Part of Trainset")
    
        test_data_=np.array(test_data_)
        test_labels_=np.array(test_labels_)
        test_labels_org_=np.array(test_labels_org_)
        train_data_=np.array(train_data_)
        train_labels_=np.array(train_labels_)
        train_labels_org_=np.array(train_labels_org_)
        
        
        print('data ready')
        return [test_data_,  test_labels_, train_data_, train_labels_, test_labels_org_, train_labels_org_]
    
#    def goodfactors(self):
#        if self.data ==[[],[],[],[]]:
#            self.data=self.readData(self.path, self.trainbasename, self.testbasename)
#        train_data=(self.data[2])
#        # Test data is *not* used when calculating the mean and std.
#        #mean = 0#train_data.mean(axis=0)
#        std = train_data.std(axis=0)
#        #train_data_ = (train_data_ - mean) / std 
#        #test_data_ = (test_data_ - mean) / std   
#        print('std should be 1. This is how std looks: ',std)
        

    def build_model(self):

        winit=keras.initializers.RandomNormal()
        #winit=keras.initializers.RandomUniform(minval=0, maxval=0.1)
        
        #import matplotlib.pyplot as plt
        #print(tf.__version__)
        NonNeg=keras.constraints.NonNeg()
        #NonNeg=keras.constraints.MaxNorm(max_value=50, axis=0)
        
        inputs=len(self.prepareData(([0]*self.nr_n),([0]*self.nr_u)))#(self.data[2].shape[1],)
        inputs=(inputs,)
        model = keras.Sequential()
            # we can think of this chunk as the input layer
        model.add(keras.layers.Dense(32, input_shape=inputs,kernel_constraint=NonNeg,kernel_initializer=winit))
        model.add(keras.layers.Activation('hard_sigmoid'))
        
        model.add(keras.layers.Dense(32,kernel_constraint=NonNeg,kernel_initializer=winit))
        model.add(keras.layers.Activation('hard_sigmoid'))

        model.add(keras.layers.Dense(1,kernel_constraint=NonNeg,kernel_initializer=winit))
    
        optimizer = tf.train.RMSPropOptimizer(0.001, centered=False, momentum=0.8)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model
    
    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('',epoch)
        print('.', end='')
    
    
    def plot_history(self):
        history=self.history_T
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        epochOld=0
        for h in history:
          plt.plot([x+epochOld for x in h.epoch], np.array(h.history['mean_absolute_error']), 
                   label='Train Loss')
          plt.plot([x+epochOld for x in h.epoch], np.array(h.history['val_mean_absolute_error']),
                   label = 'Val loss', alpha=0.5)
          plt.legend()
          epochOld+=max(h.epoch)
          #plt.ylim([0,5])
          
    def plot3d(self, dataload):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        ax.set_xlabel('PWM [3000=100%]')
        ax.set_ylabel('n [rad/s]')
        ax.set_zlabel('Tz [Nm]')
        
        if dataload:
            data_ =self.data
            
            test_data=(data_[0])
            #test_labels=(data_[1])
            train_data=(data_[2])
            #train_labels=(data_[3])
            #original torque
            test_labels_org=(data_[4])
            train_labels_org=(data_[5])
            #plot 3d test and train  data
            n=[row[0] for row in train_data]
            u=[row[1] for row in train_data]
            n_test=[row[0] for row in test_data]
            u_test=[row[1] for row in test_data]
            
            #acc_smooth_test=[row[5] for row in test_data]
            #acc_smooth_train=[row[5] for row in train_data]
            
            #plot with speed
            ax.plot(np.array(u)*self.pwm_factor, np.array(n)*self.n_factor, train_labels_org, ',', c='b') #,=pixel
            ax.plot(np.array(u_test)*self.pwm_factor, np.array(n_test)*self.n_factor, test_labels_org, ',', c='r') #,=pixel
             #plot delta with speed
            #ax.plot(np.array(u)*self.pwm_factor, np.array(n)*self.n_factor, train_labels, ',', c='b') #,=pixel
            #ax.plot(np.array(u_test)*self.pwm_factor, np.array(n_test)*self.n_factor, test_labels, ',', c='r') #,=pixel
           
            
            
            #plot delta with acceleration
            #ax.plot(np.array(u)*self.pwm_factor, np.array(acc_smooth_train)*self.n_factor, train_labels, ',', c='b') #,=pixel
            #ax.plot(np.array(u_test)*self.pwm_factor, np.array(acc_smooth_test)*self.n_factor, test_labels, ',', c='r') #,=pixel

        # create x,y
        
        xx, yy =np.meshgrid(np.linspace(-3000,3000, num=50, endpoint=True), np.linspace(-5,5, num=50, endpoint=True))
        datafunk=[]
        z_linear=[]
        for i in range(len(xx)):
            dap=[]
            z_linear_i=[]
            for j in range(len(xx[i])):
                #datum=[xx[j][i], yy[j][i]]
                datum= self.prepareData(np.array([yy[i][j]]*self.nr_n),np.array([xx[i][j]]*self.nr_u))
                dap.append(datum)
                z_linear_i.append(self.linearModel_dcx22(yy[i][j],xx[i][j]))
                #if abs(z_linear_i[-1])>2:
                #    z_linear_i[-1]=np.nan
            datafunk.append(dap)
            z_linear.append(z_linear_i)
        z_model=[]
        for dap in datafunk:
            z_model.append(self.model_T.predict(np.array(dap)).flatten())
        z=[]
        
        for i in range(len(z_linear)):
            zi=z_linear[i]+z_model[i]
            for j in range(len(zi)):
                if abs(zi[j])>5:
                    zi[j]=np.nan
            for j in range(len(z_linear[i])):
                if abs(z_linear[i][j])>5:
                    z_linear[i][j]=np.nan
            z.append(zi)
            
        #for zl, zm in zip(z_linear, z_model):
        #    z.append(zl+zm)
        ax.plot_surface(xx,yy, np.array(z), alpha=0.5,color='b')
        #ax.plot_surface(xx,yy, np.array(z_linear), alpha=0.5,color='r')
        #ax.plot_surface(xx,yy, np.array(z_model), alpha=0.5,color='g')
        
        return fig
        
    def plot_model(self):
        
        if self.data ==[[],[],[],[]]:
            self.data=self.readData()
            
        import matplotlib.pyplot as plt

        data_ =self.data
        
        test_data=(data_[0])
        test_labels=(data_[1])
        #train_data=(data_[2])
        #train_labels=(data_[3])
        test_labels_org=(data_[4])
        test_linear_model=[]
        
        test_predictions = self.model_T.predict(test_data).flatten()
        test_predictions_org=[]
        for i in range(len(test_predictions)):
            test_linear_model.append( self.linearModel_dcx22(test_data[i][0]*self.n_factor,test_data[i][1]*self.pwm_factor))
            test_predictions_org.append(test_predictions[i] + test_linear_model[i])
        #print(test_predictions)
        
        plt.figure()
        plt.plot(test_labels, label="delta_Tz(test)[Nm]",  alpha=0.5,color='C2')
        plt.plot(test_predictions, label="delta_Tz_calc(test)[Nm]",  alpha=0.5,color='C3')
        plt.plot(test_predictions_org,label="Tz_calc(test)[Nm]", color='C1')
        plt.plot(test_labels_org,label="Tz(test)[Nm]", color='C0')

        #plt.plot(test_linear_model,label="Tz_calc_linear(test)[Nm]")
        
        #plot 2d test data
        n_test=[row[0] for row in test_data]
        u_test=[row[1] for row in test_data]
        
        n_mean_test=[row[2] for row in test_data]
        #if self.acceleration:
        #    acc_smooth_test=[row[5] for row in test_data]
        #powerd_test=[row[len(test_data[0])-1] for row in test_data]
        pwmXn_test=[row[len(test_data[0])-2] for row in test_data]
        
        
        plt.plot(np.array(n_test), label="n(test)[rad/s]", color = 'C4',alpha=0.5)
        if self.n_mean>1:
            plt.plot(np.array(n_mean_test),  alpha=0.5, label="n_mean(test)")
        plt.plot(np.array(u_test), label="u(test)[PWM/PWM_max]",color = 'C5' ,alpha=0.5)
        #if self.acceleration:
        #    plt.plot(acc_smooth_test, alpha=0.1) #, label="acc(test)"
        if self.driven:
            plt.plot(pwmXn_test, alpha=0.1) #label='pwmXn_direction(test)', 
        #if self.power_direction:
        #    plt.plot(powerd_test,  alpha=0.1) #,label='Power_direction(test)'
        plt.legend()
        
        self.plot3d(True)
        
        #Let's see how did the model performs on the test set:
        [loss, mae] = self.model_T.evaluate(test_data, test_labels, verbose=0)
        print("\n Testing set Mean Abs Error: {:7.4f}".format(mae))
        mpe_calc=mae/np.mean(np.absolute(np.array(test_labels_org)))
        
        mae_linear=np.mean(abs(np.array(test_linear_model)-np.array(test_labels_org)))
        print("\n Testing set pertantage Error (own calc): {:7.4f}".format(mpe_calc))
        print("\n Testing set Mean Abs Error: (Linear Model){:7.4f}".format(mae_linear))
        
    def plot_slice(self,n):
        T_model=[]
        T_linear=[]

        dap=[]
        pwms=[]
        for pwm in np.linspace(-3000,3000, num=50, endpoint=True):#range(-3000,3000,1):
            pwms.append(pwm)
            datum= self.prepareData(np.array([n]*self.nr_n),np.array([pwm]*self.nr_u))
            dap.append(datum)
            T_linear.append(self.linearModel_dcx22(n,pwm))
            
        T_model=(self.model_T.predict(np.array(dap)).flatten()) 
        T_modelLin=[T_linear[i]+T_model[i] for i in range(len(T_model))]
            
        import matplotlib.pyplot as plt
        plt.figure()        
        
        plt.plot(pwms,T_model,label="Model")
        plt.plot(pwms,T_linear,label="linear")
        plt.plot(pwms,T_modelLin,label="Model+linear")

        
        plt.xlabel('PWM Value')
        plt.ylabel('T')
        plt.legend()
        
    def train_model(self, EPOCHS_):  
        
        if self.data ==[[],[],[],[]]:
            self.data=self.readData()
            
        import numpy as np
        data=self.data
        model_T=self.model_T
        test_data_=data[0]
        test_labels_=data[1]
        train_data_=data[2]
        train_labels_=data[3]
        # Shuffle the training set
        order = np.argsort(np.random.random(train_labels_.shape))
        train_data_ = train_data_[order]
        train_labels_= train_labels_[order]
    
        #the labels are the Torques in Nm
        #print("train_labels[0:10]:",train_labels_[0:10])  # Display first 10 entries
        #print("train_data[0:10]:",train_data_[0:10])  # Display first 10 entries
        
        print(" First training sample, normalized", train_data_[0])  # First training sample, normalized
        
        self.history_T.append(model_T.fit(train_data_, train_labels_, batch_size=2048, epochs=EPOCHS_, validation_split=0.1, verbose=0, callbacks=[self.PrintDot()]))
        
        #Let's see how did the model performs on the test set:
        [loss, mae] = model_T.evaluate(test_data_, test_labels_, verbose=0)
        
        
        
        print("\n Testing set Mean Abs Error: {:7.4f}".format(mae))
        #save model
        #text_file = open(savefilename+".json", "w")
        #text_file.write(model.to_json())
        #text_file.write('\n')
        #text_file.write("testfilename:"+testbasename)
        #text_file.write('\n')
        #text_file.write("trainfilename:"+trainbasename)
        #text_file.close()
        model_T.save_weights("./"+self.netpath+self.savefilename+".h5",overwrite=True)
        #PLOT THE TRAINING HISTORY
        self.plot_history()
    
    def predict(self,n,pwm):
        #input with n[0] oldest and n[-1] the newest
        dap=self.prepareData(np.array(n)/self.n_factor,np.array(pwm)/self.pwm_factor)
        Tz=self.model_T.predict(np.array([dap])).flatten()
        Tz=Tz+self.linearModel_dcx22(n[-1],pwm[-1])
        return Tz, self.zukunft
        #print('predict')

    def savedatatofile(self):
        import pickle
        datasave=self.data
        with open("datasave.pickle", 'wb') as f:
            pickle.dump(datasave, f)
            
    def loaddatafromfile(self):
        import pickle         
        with open("datasave.pickle", 'rb') as f:
            self.data = pickle.load(f)
"""   
#use:
nnpredict=NNPredict()
#use class
import class_nn_T_prediction_xx as nnp
nnpredict=nnp.NNPredict()
#predict
nnpredict.predict(n[-5:-1],pwm[-5:-1])
 
    
#Train the Model and  Store training stats
nnpredict=NNPredict()
nnpredict.train_model( EPOCHS)
#Let's see how did the model performs on the test set:
    
nnpredict.plot_model()
"""
    

