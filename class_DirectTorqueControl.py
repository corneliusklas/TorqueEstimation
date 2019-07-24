# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

import class_nn_T_prediction_15_Delta_simple as nnp

#from mpl_toolkits.mplot3d import Axes3D

# Argumentwerte als 1D Arrays erzeugen
#n_1d = np.linspace(-13000,13000,301)
#u_1d = np.linspace(-48,48,301)

TOL = 0.01
MAXN=10

class c_DTC:
    nnpredict=nnp.NNPredict()
    eI=0
    e_old=0
    def control(self,T_soll,n):
        du=10
        u=self.linearModel_dcx22([[T_soll,n]])
        u1=u+du
        u2=u-du
        
        f=self.nnpredict.predict #nnpredict.predict([n],[pwm])[0][0]
        u= self.regulaFalsi(f,T_soll,n,u1,u2,TOL,MAXN)
        #print ('eP',eP, 'eD',eD)
        return u
    

    
    def regulaFalsi(self,f_,T_soll,n,u1,u2,TOL,N,printtext=False):
        
        def f(u):
            return f_([n],[u])[0][0]-T_soll
        
        i = 1
        Fu1 = f(u1)
         
        if printtext: print("%-20s %-20s %-20s %-20s %-20s" % ("n","u1_n","u2_n","p_n","f(p_n)"))
          
        while(i <= N):
            p = (u1*f(u2)-u2*f(u1))/(f(u2) - f(u1))
            FP = f(p)
              
            if(FP == 0 or np.abs(f(p)) < TOL):
                if printtext: print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, u1, u2, p, f(p)))
                break
            else:
                 if printtext: print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, u1, u2, p, f(p)))
             
              
            i = i + 1
              
            if(Fu1*FP > 0):
                u1 = p
            else:
                u2 = p
        return p
    
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

        T=np.array([row[0] for row in Tn])
        n=np.array([row[1] for row in Tn]) *n_factor
        T_motor=T*T_factor/Motor_eta
        #U(M,n)=(n-b*M)/a
        U=(n-motor_b*T_motor)/motor_a

        pwm=(U-U_lost*np.sign(U))/Umax*(pwmmax-pwm_zero)+pwm_zero*np.sign(U) 

        return pwm
#
#def Motor(n,u):
#    #https://www.maxonmotor.de/maxon/view/catalog/
#    eta =0.5
#    i=200
#    a=226 #1/min/V
#    b=123000 #1/min/Nm
#    #n= a*U
#    T= (n+a*u)/b
#    T=T*i*eta
#    try:
#        for i in range(len(T)):
#            for j in range(len(T[i])):
#                if abs(T[i][j])>2:
#                    T[i][j]=np.sign(T[i][j])*2
#    except:
#        if abs(T)>2:
#            T=np.sign(T)*2
#    #print(T)
#    return T
#
## Argumentwerte als 2D Arrays erzeugen
##n_2d, u_2d = np.meshgrid(n_1d, u_1d)
#
## Interessante Daten erzeugen
##z_2d = Motor(n_2d,u_2d)#1/(x_2d**2 + y_2d**2 + 1) * np.cos(np.pi * x_2d) * np.cos(np.pi * y_2d)
#
## Plotten
#pid1=c_PID()
#
#fig,ax=plt.subplots()
# #gca(projection='3d')
##ax.plot_surface(n_2d, u_2d, z_2d, alpha=0.5)#cmap=plt.get_cmap("jet"))
##sc=ax.scatter(0,0,0.1,color='r')
#t=[0,0]
#n=[0,0]
#u=[0,0]
#T=[0,0]
#T_want=[0,0]
#
#ln=plt.plot(t,np.array(n)*0.0038,label='n')
#lu=plt.plot(t,u,label='u')
#lT=plt.plot(t,T,label='T')
#lT_want=plt.plot(t,T_want,label='T_want')
#
#plt.legend()
#
#ax.set_ybound(-50,50)
#ax.set_xbound(-10,0)
#
#run=True
#
#def onmove(event):
#    global x
#    global y
#    x=(event.x/fig.canvas.width()-0.5)*13000*2
#    y=(event.y/fig.canvas.height()-0.5)*2*2
#
#def handle_close(evt):
#    print('Closed Figure!')
#    global run
#    run=False
#
#clo = fig.canvas.mpl_connect('close_event', handle_close)
#cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
#
#while run:
#    #sc.remove()
#    #sc=ax.scatter(x,y,Motor(x,y),color='r')
#    n.append(x)
#    T_want.append(y)
#    T.append(Motor(n[-1],u[-1]))
#    t.append(t[-1]+1)
#    u.append(pid1.control(T_want[-1],T[-1]))
#    
#    ln[0].set_data(t,np.array(n)*0.0038)
#    lu[0].set_data(t,u)
#    lT[0].set_data(t,np.array(T)*20)
#    lT_want[0].set_data(t,np.array(T_want)*20)
#    
#    ax.set_xbound(t[-1]-50,t[-1])
#    
#    #fig.canvas.update()
#    #fig.show()
#    plt.pause(0.1)
#    #print(x, y)
#    
#    