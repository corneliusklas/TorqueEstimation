# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:15:06 2018

@author: cklas
"""
class Motordata:

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
     #print(tf.__version__

    loadData=False
    path='Data/'
    netpath='Networks/'
    
    #basename =  input('Enter file basename: ')    
    #read the data_
    n_old=0
    nr_n=n_old+1
    #new_model=False  # if new set ==False the old  model is used, but new test data or training data can be loaded
    EPOCHS = 100
    zukunft=0 # zukünftige werte für die Drehmomentschätzung - verschiebung des torque in die zukunf
    
    driven=False # das produkt aus pwm und drehzahl wird als parameter verwendet
    acceleration=True # die Beschleunigung wird als eingabeparameter verwendet

       
    trainbasename='0_M1_sum_all_Nzero_train.csv'
    testbasename='0_M1_sum_all_Nzero_test.csv'
    savefilename="NN_T_controll_7"  
    



    
    history_C=[]
    
    data=[[],[],[],[]]
    
    power_d_old=0
    
    def __init__(self):
        #self.data=self.readData(self.path, self.trainbasename, self.testbasename)

        if self.loadData:
            self.data=self.readData(self.path, self.trainbasename, self.testbasename)
    
    
    def prepareData(self,n,Tz):

        
        np=self.np
        
        n_old=self.n_old
        #datum=[n[i],u[i], n[i-1], n[i-2], n[i-3],u[i-1], u[i-2], u[i-3]]
        #this function gehts the last entries of n and u
        n=np.array(n)
        Tz=np.array(Tz)
        
        #first the last entries
        datum=[Tz[-1], n[-1]]

        #then the other entries of speed and pwm    
        for j in range( -n_old-1,-1):
            datum.append(n[j])
        for j in range( -n_old-1,-1):
            datum.append(Tz[j])
    
        if self.driven:
            print('to do')
            #power direction    : drive or driven
            #power_direction= np.sign(u[-1]*n[-1])
            #if power_direction==0:
            #    power_direction=self.power_d_old
            #else:
            #    self.power_d_old=power_direction
            #datum.append(power_direction)    
        return datum
        
    def readData(self,path, trainbasename, testbasename):
        import numpy as np
        zukunft=self.zukunft 
        #n_old=self.n_old
        #acceleration=self.acceleration
        #driven=self.driven
        
        trainAndTestData=[]
        trainAndTestLabels=[]
        
        paths=[path+trainbasename,path+testbasename]
        for train_test,filepath in enumerate(paths):
            data_ = np.genfromtxt(filepath, delimiter=',', skip_header=1, skip_footer=1, names=['name','t','Tz','n','pwm','nix'], dtype=None, encoding='utf-8')
            print('Data ',train_test, '(0:train, 1:test): ',filepath)
            
            #remove values out of sensor range for test set ownly
            if train_test==1:
                Tz_sensor_range=1
                for i in range(len(data_['Tz'])):
                    if data_['Tz'][i]>Tz_sensor_range:
                        data_['Tz'][i]=np.nan

                        
            #cut data to good range
            pwmmax=2999
            nmax=1000
            for i in range(len(data_)):
                if abs(data_['pwm'][i])>pwmmax:
                    data_['pwm'][i]=np.nan
                if abs(data_['n'][i])>nmax:
                    data_['n'][i]=np.nan
                
                #if abs(data_['n'][i])>0 and data_['pwm'][i]<0 :
                #    data_['n'][i]=np.nan
                      
            
            #seperate interesting data
            if zukunft>0:
                trainOrTestLabels=data_['pwm'][self.nr_n-zukunft:-zukunft] 
            else:
                trainOrTestLabels=data_['pwm'][self.nr_n:] 
            trainOrTestData=[]
            n=data_['n']#/460
            Tz=data_['Tz']
            

            
            for i in range(self.nr_n,len(data_)): 
                datum= self.prepareData(n[i-self.nr_n:i+1],Tz[i-self.nr_n:i+1])
                trainOrTestData.append(datum)
                
            #remove values out of sensor range = NAN VALUES
            trainOrTestLabels=list(trainOrTestLabels)
            trainOrTestData=list(trainOrTestData)
            trainOrTestLabels_cut=[]
            trainOrTestData_cut=[]
            
            
            #check for nan values
            for i in range(len(trainOrTestLabels)):
                if np.isnan(trainOrTestLabels[i])==False and np.isnan(sum(trainOrTestData[i]))==False:
                    trainOrTestLabels_cut.append(trainOrTestLabels[i])
                    trainOrTestData_cut.append(trainOrTestData[i])
                    
            trainOrTestData =   trainOrTestData_cut
            trainOrTestLabels=trainOrTestLabels_cut 
            
 
            trainAndTestData.append(trainOrTestData)
            trainAndTestLabels.append(trainOrTestLabels)
    
        test_data_=(trainAndTestData[1])
        test_labels_=(trainAndTestLabels[1])
        train_data_=(trainAndTestData[0])
        train_labels_=(trainAndTestLabels[0])
        
        
        print('data ready')
        return [test_data_,  test_labels_, train_data_, train_labels_]
    
    def linearModel_dcx22(self,Tn):
            #ticks/ms -> U/min
        n_factor=1000*60/4096
        #übersetzung getriebe + Nm->m
        T_factor=1000/231
        #Drehzahlkonstante   [min-1 V-1] 
        motor_a	= 226 +35
        #Kennliniensteigung [min-1 mNm-1] 
        motor_b=123
        #Wirkungsgrad Getriebe *Motor	
        Motor_eta= 0.4#0.75*0.85
        #verlust im pwm schaltkreis
        U_lost=0 #3
        #Spannung bei pwm max
        Umax=48#+U_lost

        pwmmax=3000
        pwm_zero=250 #400

        T=self.np.array([row[0] for row in Tn])
        n=self.np.array([row[1] for row in Tn]) *n_factor
        T_motor=T*T_factor/Motor_eta
        #U(M,n)=(n-b*M)/a
        U=(n-motor_b*T_motor)/motor_a

        pwm=(U-U_lost*self.np.sign(U))/Umax*(pwmmax-pwm_zero)+pwm_zero*self.np.sign(U) 

        return pwm
    
    def linearModel_dcx16(self,Tn):
            #ticks/ms -> U/min
        n_factor=1000*60/4096
        #übersetzung getriebe + Nm->m
        T_factor=1000/406
        #Drehzahlkonstante   [min-1 V-1] 
        motor_a	= 265
        #Kennliniensteigung [min-1 mNm-1] 
        motor_b=620
        #Wirkungsgrad Getriebe *Motor	
        Motor_eta= 0.55*0.78

        #verlust im pwm schaltkreis
        U_lost=2
        #Spannung bei pwm max
        Umax=48
        pwmmax=3000
        pwm_zero=400

        T=self.np.array([row[0] for row in Tn])
        n=self.np.array([row[1] for row in Tn]) *n_factor
        T_motor=T*T_factor/Motor_eta
        #U(M,n)=(n-b*M)/a
        U=(n-motor_b*T_motor)/motor_a

        pwm=(U-U_lost*self.np.sign(U))/Umax*(pwmmax-pwm_zero)+pwm_zero*self.np.sign(U)
        return pwm
    
    def linearModel_dcx22_T(self,n,pwm):

        #ticks/ms -> U/min
        n_f=1000*60/4096
        #übersetzung getriebe + Nm->m
        T_f=1000/231
        #Drehzahlkonstante   [min-1 V-1] 
        motor_a	= 226+30
        #Kennliniensteigung [min-1 mNm-1] 
        motor_b=123
        #Wirkungsgrad Getriebe *Motor	
        Motor_eta= 0.4# max from datasheet:0.75*0.85
        #verlust im pwm schaltkreis
        U_lost=0
        #Spannung bei pwm max
        Umax=48#+U_lost

        pwmmax=3000
        pwm_zero=250
        
        n=n *n_f
        U=max(0,(abs(pwm)-pwm_zero)/(pwmmax-pwm_zero)*(Umax-U_lost))*np.sign(pwm)

       
        #U(M,n)=(n-b*M)/a
        T_motor=(U*motor_a-n)/-motor_b
        T=T_motor*Motor_eta/T_f
            
        return T
    def plot_safetyborder(self):
        import matplotlib.pyplot as plt
        Tmax_dcx16=2.5 # [Nm]
        Tmax_dcx22=6.2 # [Nm]
        
        Tn16=[]
        Tn22=[]
        Tn16_=[]
        Tn22_=[]
        
        for ni in range(-1000,1000,1):
            Tn16.append([Tmax_dcx16,ni])
            Tn22.append([Tmax_dcx22,ni])
            Tn16_.append([-Tmax_dcx16,ni])
            Tn22_.append([-Tmax_dcx22,ni])
        
        pwmmax16=self.linearModel_dcx16(Tn16)
        pwmmax22=self.linearModel_dcx22(Tn22)
        pwmmin16=self.linearModel_dcx16(Tn16_)
        pwmmin22=self.linearModel_dcx22(Tn22_)
        
        n=[row[1] for row in Tn16]
        
        plt.plot(n,pwmmax16,label="PWM max DCX 16")
        plt.plot(n,pwmmax22,label="PWM max DCX 22")
        plt.plot(n,pwmmin16,label="PWM min DCX 16")
        plt.plot(n,pwmmin22,label="PWM min DCX 22")
        
        plt.ylabel('PWM Safety border')
        plt.xlabel('n [Ticks /ms]')
        
        plt.legend()
    
    def plot_model(self):
        
        if self.data ==[[],[],[],[]]:
            self.data=self.readData(self.path, self.trainbasename, self.testbasename)
            
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        data_ =self.data
        
        test_data=np.array(data_[0])
        test_labels=np.array(data_[1])
        train_data=np.array(data_[2])
        train_labels=np.array(data_[3])
        
        test_predictions=self.linearModel_dcx22(test_data)

        plt.figure()
        plt.plot(test_labels, label="test_PWM")
        plt.plot(test_predictions, label="test_PWM_calc")
        plt.ylabel('Normalised Speed/Torque\n PWM/1000')
        plt.xlabel('Timestep')
        
        #plot 2d test data
        n_test=[row[1] for row in test_data]
        T_test=[row[0] for row in test_data]
        #powerd_test=[row[len(test_data[0])-1] for row in test_data]
        
        plt.plot(n_test, label="n_test")
        plt.plot(T_test, label="T_test")
        #plt.plot(powerd_test,label='Power_direction_test')
        plt.legend()
        
        #plot 3d test and train  data
        n=[row[1] for row in train_data]
        T=[row[0] for row in train_data]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        ax.plot(np.array(T), np.array(n), train_labels, ',', c='b') #,=pixel
        ax.plot(np.array(T_test), np.array(n_test), test_labels, ',', c='r') #,=pixel
        ax.set_xlabel('tz[Nm]')
        ax.set_ylabel('n[Ticks/ms]')
        ax.set_zlabel('pwm')
        
        

        # create x,y
        
        #xx=tz, yy=n
        xx, yy =np.meshgrid(np.linspace(-2,2, num=50, endpoint=True), np.linspace(-1000,1000, num=50, endpoint=True))
        datafunk=[]
        for i in range(len(xx)):
            dap=[]
            for j in range(len(xx[i])):
                datum=[xx[i][j], yy[i][j]]
#                datum= self.prepareData(np.array([yy[i][j]]*(self.nr_n+1)),np.array([xx[i][j]]*(self.nr_n+1)))
#                for k in range( self.n_old):
#                    datum.append(xx[j][i])
#                for k in range( self.n_old):
#                    datum.append(yy[j][i])
#                #power_direction
#                if self.driven: 
#                    datum.append(np.sign(xx[j][i]* yy[j][i]))
                dap.append(datum)
#                #dap.append([xx[i][j],yy[i][j]]) 
            datafunk.append(dap)
        z=[]
        for dap in datafunk:
            #print(dap)
            z.append(self.linearModel_dcx22(dap))
        ax.plot_surface(xx,yy, np.array(z), alpha=0.5)
        

        
md =Motordata()
#md.plot_model()
md.plot_safetyborder()