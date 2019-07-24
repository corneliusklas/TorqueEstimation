# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:35:02 2018

@author: cklas
"""

#!/usr/bin/python3
 
# adapted from https://github.com/recantha/EduKit3-RC-Keyboard/blob/master/rc_keyboard.py
 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import class_nn_T_prediction_15_Delta_simple as nnp
nnpredict=nnp.NNPredict()

import class_DirectTorqueControl as dtc
dtc1=dtc.c_DTC()

v3d=False
#plt.axis([-5, 5, -5, 5])

#for i in range(10):
#    y = np.random.random()
#    plt.scatter(i, y)
#    plt.pause(0.05)

#plt.show()
n=[0,0,0,0,0]
n_i=0
T_wanted=[0,0,0,0,0]
T_wanted_i=0
T_calc=[0,0,0,0,0]
T_in=[0,0,0,0,0]
#T_calc_i=0
pwm_i=0
pwm=[0,0,0,0,0]
t=[0,0,0,0,0]
dt=0.01
running=True
v_exit = False
v_train=False

#plot variables
ax_T=0
ax_u=0
ax_n=0
lT_wanted=0
lT=0
ln=0
lu=0

def press(event):
    print('press', event.key)
    sys.stdout.flush()
    global running
    global n_i
    global T_wanted_i
    global pwm_i
    global T_wanted
    global pwm
    global nncontrol
    global T_calc
    global T_in
    global n
    global t
    global v_exit    
    global v_train
    
    if event.key == 'x':
        running=False
        v_exit=True
        #plt.close('all')
    if event.key == 'p':
        running=not(running)
    elif event.key == 'up':    
        n_i+=10
    elif event.key == 'down': 
        n_i-=10
    elif event.key == 'right':    
        T_wanted_i+=0.1
        #pwm_i+=10
    elif event.key == 'left': 
        T_wanted_i-=0.1
        #pwm_i-=10
    elif event.key=='r':
        prepareplot()
        running=True
        n=[0,0,0,0,0]
        n_i=0
        T_wanted=[0,0,0,0,0]
        T_wanted_i=0
        T_calc=[0,0,0,0,0]
        pwm_i=0
        pwm=[0,0,0,0,0]
        t=[0,0,0,0,0]
    #change pid settings
#    elif event.key=='v':
#        pid1.TD=pid1.TD*1.1
#        print('pid1.TD:'+str(pid1.TD))
#    elif event.key=='b':
#        pid1.TD=pid1.TD*0.9
#        print('pid1.TD:'+str(pid1.TD))
#    elif event.key=='n':
#        pid1.TI=pid1.TI*1.1
#        print('pid1.TI:'+str(pid1.TI))
#    elif event.key=='m':
#        pid1.TI=pid1.TI*0.9
#        print('pid1.TI:'+str(pid1.TI))
#    elif event.key==',':
#        pid1.KP=pid1.KP*1.1
#        print('pid1.KP:'+str(pid1.KP))
#    elif event.key=='.':
#        pid1.KP=pid1.KP*0.9
#        print('pid1.KP:'+str(pid1.KP))

def moved(event):
    global n_i
    global T_wanted_i
    #global pwm_i
    
    if event.ydata != None:
        n_i = event.ydata#/(ax_n.get_xbound()[1]-ax_n.get_xbound()[0])
    if event.xdata != None:
        T_wanted_i = (event.xdata-ax_n.axes.get_xlim()[0])/(ax_n.axes.get_xlim()[1]-ax_n.axes.get_xlim()[0])*1.5-0.75
        #print(event.xdata, pwm_i)
    
    
def close(event):
    global running
    global v_exit
    running=False
    v_exit = True

  
#v3d plot    
if v3d:
    fig3d=nnpredict.plot3d(False)
    ax3d=fig3d.axes[0]

def prepareplot():
    global ax_T
    global ax_u
    global ax_n
    global lT_wanted
    global lT
    global lT_in
    global ln
    global lu
    #generate Torque subplot
    fig, ax_T = plt.subplots()
    
    #fullscreen
    mng = plt.get_current_fig_manager()
    #mng.frame.Maximize(True) ubuntu
    mng.window.showMaximized()
    
    
    ax_T.tick_params('y', colors='m', pad=40)
    ax_T.yaxis.tick_left()
    ax_T.yaxis.set_label_position('left')
     #generate subplot with twin x axes - PWM
    ax_u = ax_T.twinx()
    ax_u.tick_params('y', colors='r', pad=40)
    ax_u.yaxis.tick_right()
    ax_u.yaxis.set_label_position('right')
    #generate subplot with twin x axes - n
    ax_n = ax_T.twinx()
    ax_n.tick_params('y', colors='green')
    
    
    
    fig.canvas.mpl_connect('key_press_event', press)
    fig.canvas.mpl_connect('close_event', close)
    fig.canvas.mpl_connect('motion_notify_event', moved)
    
    #ax.plot(p[0],p[1])
    #xl = ax.set_xlabel('FPS')
    #ax.set_title('Press a key')
    #ax_T.clear()
    #ax_n.clear()
    #ax_u.clear()
    ax_T.axes.set_ylim([-1.5,1.5])#axis([0, 10, -1.5, 1.5])
    ax_T.set_ylabel('T [Nm]', color='m', labelpad=-10)
    ax_u.axes.set_ylim([ -3000, 3000])
    ax_u.set_ylabel('PWM [-]', color='r', labelpad=-5)
    ax_n.axes.set_ylim([ -5, 5])
    ax_n.set_ylabel('n [rad/s]', color='green',labelpad=-5)
    
    
        
    #fig.clear()
    lT_wanted=ax_T.plot(t,T_wanted,color='b',label='T_wanted')
    lT=ax_T.plot(t,T_calc,color='m',label='T_calc')
    lT_in=ax_T.plot(t,T_in,color='y',label='T_inertia')
    ln=ax_n.plot(t,n,color='green',label='n')
    lu=ax_u.plot(t,pwm,color='r',label='pwm')
    
    fig.legend()
    
    #if v3d:
    #    sc3d=ax3d.scatter([0],[0],[0])
        
    plt.ion()

prepareplot()

while not(v_exit):
    plt.pause(0.01)
    if running == True:
        #wait forinteraction
        t.append(t[-1]+dt)
        max_dt=2
         
        n.append(n_i)
        
        T_wanted.append(T_wanted_i)
        #p[1]+=0.1
        
        #calculate the control pwm
        pwm_i=dtc1.control(T_wanted[-1],n[-1])
        
        pwm.append(pwm_i)
        
        T_calc_i,zukunft=nnpredict.predict(n[-6:-1],pwm[-6:-1])
        T_calc.append(T_calc_i[0])
        T_in.append(T_calc[-1]+nnpredict.inertia_dcx22(n[-1],n[-2],0.1))
        print(T_in[-1])
        #if zukunft>0:
        #    T_calc[-zukunft-1]=T_calc_i
        
        if t[-1]>max_dt:
            ax_n.axes.set_xlim(t[-1]-max_dt, t[-1])
        else:
            ax_n.axes.set_xlim(0, t[-1])
        
    
        lT[0].set_data(t,T_calc)
        lT_wanted[0].set_data(t,T_wanted)
        ln[0].set_data(t,n)
        lu[0].set_data(t,pwm)
        lT_in[0].set_data(t,T_in)
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        
        #if v3d:
        #    sc3d.remove()
        #    sc3d=ax3d.scatter([pwm_i],[n_i],[T_calc_i])
        #    ax3d.scatter([pwm_i],[n_i],[T_calc_i])
        #    fig3d.canvas.draw()
        #    fig3d.canvas.flush_events()
    
    #plt.show()
    #running=False
    elif  v_train :
        nncontrol.train_model(200)
        v_train=False
        running=True
        prepareplot()
    else:
        print(v_train,running)
        
        
    

