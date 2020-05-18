# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:51:06 2020

@author: tonyc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cbook as cbook
import math as mth
import scipy.signal as sy
import scipy.stats as ss
import random as random
from matplotlib.patches import PathPatch
import seaborn as sns
from mpl_toolkits import mplot3d

import pandas as pd
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import time
import mpl_toolkits.mplot3d.axes3d as p3
import os
import subprocess
import matplotlib.animation as mani
from matplotlib.animation import PillowWriter

def parameters(lat,Y,M,O,beta_y,beta_m,beta_o,delta,red,h,trans):
    N_lat = lat
    beta_vec = np.zeros((6,))
    beta_vec[0] = np.float(beta_y)
    beta_vec[1] = np.float(beta_m)
    beta_vec[2] = np.float(beta_o)
    beta_vec[3] = delta
    beta_vec[4] = red
    beta_vec[5] = h
    
    N_trans = trans
    n_pop = np.zeros((3,))
    n_pop[0] = Y #Young age people
    n_pop[1] = M #Middle age'
    n_pop[2] = O #old aged
    return  N_lat, n_pop, beta_vec, N_trans

def parameters_infection(PY,PM,PO,T_s,T_i,PYSI,PMSI,POSI, N_time, pop_step):
    prob_par = np.zeros((8,))
    
    prob_par[0] = PY
    prob_par[1] = PM
    prob_par[2] = PO
    prob_par[3] = T_s
    prob_par[4] = T_i
    prob_par[5] = PYSI
    prob_par[6] = PMSI
    prob_par[7] = POSI
    return prob_par, N_time, pop_step
def more_parameters(Ts_min,Ts_max,Ti_min,Ti_max,pop_size,Ts_avg,Ti_avg,Ts_scale,Ti_scale, \
 Tawa_min,Tawa_max,Tawa_avg,Tawa_scale):
    inf_par = np.zeros((13,))
    inf_par[0] = Ts_min
    inf_par[1] = Ts_max
    inf_par[2] = Ti_min
    inf_par[3] = Ti_max
    inf_par[4] = pop_size
    inf_par[5] = Ts_avg
    inf_par[6] = Ti_avg
    inf_par[7] = Ts_scale
    inf_par[8] = Ti_scale
    inf_par[9] = Tawa_min
    inf_par[10] = Tawa_max
    inf_par[11] = Tawa_avg
    inf_par[12] = Tawa_scale
    
    
    return inf_par
def random_side(N_lat):
    
    val_1 = random.randint(0,2)
    if val_1 == 0:
        posx = random.randint(0,N_lat-1)
        posy = random.randint(0,N_lat-1)
        posz = random.randint(0,1)*(N_lat-1)
    if val_1 == 1:
        posx = random.randint(0,N_lat-1)
        posy = random.randint(0,1)*(N_lat-1)
        posz = random.randint(0,N_lat-1)
    if val_1 == 2:
        posx = random.randint(0,1)*(N_lat-1)
        posy = random.randint(0,N_lat-1)
        posz = random.randint(0,N_lat-1) 
    return posx,posy,posz

def fill_lattice(lattice, n_pop, N_lat):
    lattice_3d = lattice
    young_s = 0
    middle_s = 0
    old_s = 0
    fill = 0
    is_filled = False
    while (not is_filled):     
        fill +=1
        #i,j,k = random_side(N_lat)
        i,j,k = random_location(N_lat)
        if lattice_3d[i,j,k] > 0:
            continue
        else:
            lattice_3d[i,j,k] = 1 
            young_s += 1
        if young_s == n_pop[0]:
            is_filled = True
            break
    
 
    is_filled = False    
    while (not is_filled):

        i,j,k = random_location(N_lat)
        if lattice_3d[i,j,k] > 0:
            continue
        else:
            lattice_3d[i,j,k] = 5 
            middle_s += 1
        if middle_s == n_pop[1]:
            is_filled = True
            break
    
    is_filled = False
    while( not is_filled):
        i,j,k = random_location(N_lat)
        if lattice_3d[i,j,k] > 0:
            continue
        else:
            lattice_3d[i,j,k] = 9
            old_s += 1
        if old_s == n_pop[2]:
            is_filled = True
            break
    
    return lattice_3d
# Choose a random location
def random_location(N_lat):

    posx = random.randint(0,N_lat-1)
    posy = random.randint(0,N_lat-1)
    posz = random.randint(0,N_lat-1)

    return posx,posy,posz


# Randomly choose another spot adjacent
def random_movement(i,j,k,size):
    
        
    posx = i + random.randint(-1, 1)
    posy = j + random.randint(-1, 1)
    posz = k + random.randint(-1, 1)
        
    posx = posx % size
    posy = posy % size
    posz = posz % size
    return posx,posy,posz

def nnPBC_sum(i,j,k,lattice,x,y,z, size,beta,age):
    
    #parameters

    sum_lat = 0.0 
    # x,y,z represents previous spot

    is_adjacent_nx_spt = False
  #  age_y, age_m, age_o = check_age_specific(x,y,z,lattice, age) 
    #print(i,j,k, "Start of each loop")
    ii_ran = range(i-1,i+2)
    jj_ran = range(j-1,j+2)
    kk_ran = range(k-1,k+2)
    for ii in ii_ran:  
       iii = ii % size
       for jj in jj_ran:           
           jjj = jj % size
           for kk in kk_ran:                                         
               kkk = kk % size
                     
               spot_check = lattice[iii,jjj,kkk]
               if ([iii, jjj, kkk] == [i,j,k]) or ([iii, jjj, kkk] == [x,y,z]) or (spot_check == 0):
                   #print(ii,jj,kk, "b")
                   continue
               else:
                   if spot_check in [3,7,11]:
                       is_adjacent_nx_spt = True
                       continue
                 
                   if spot_check in age:
                       #print("working")
                       sum_lat = sum_lat + beta*1.0


    return sum_lat, is_adjacent_nx_spt


def Population_equilibrate(lattice, N_lat, beta_vec, N_trans):
    b_y = beta_vec[0]
    b_m = beta_vec[1]
    b_o = beta_vec[2]
    
    for qq in range(0,N_trans):

#        is_Outside = False
        i,j,k =  random_location(N_lat)#randomly_choose_an_agent(lattice, N_lat)
        if (lattice[i,j,k] == 0):
            continue        
        m,n,o = random_movement(i,j,k,N_lat)
        if lattice[m,n,o] > 0:
            continue
        age1 = lattice[i,j,k]
        if (1 <= age1 <= 4):
            age = [1,2,4]
            beta = b_y
        elif (5 <= age1 <= 8):
            age = [5,6,8]
            beta = b_m
        else:
            age = [9,10,12]
            beta = b_o
        N_1, is_adjacent = nnPBC_sum(i, j, k, lattice, i,j,k,N_lat,beta,age) 

        N_2, is_adjacent  = nnPBC_sum(m,n,o,lattice,i,j,k, N_lat,beta,age) 
       
        dE = N_2 - N_1
        
        prob_n = np.exp(dE)
        rand_val =  random.random()
        if  rand_val < prob_n:
            # moved with probability e^(beta(delatN))
            lattice[m,n,o] = lattice[i,j,k] 
            lattice[i,j,k] = 0

# =============================================================================
#         xany,yany,zany = (lattice==1).nonzero()
#         xanm,yanm,zanm = (lattice==5).nonzero()
#         xano,yano,zano = (lattice==9).nonzero()
#         ax.scatter(xany,yany,zany,c='red',marker="o")
#         ax.scatter(xanm,yanm,zanm,c='blue',marker="o")
#         ax.scatter(xano,yano,zano,c='black',marker="o")
#         plt.draw()
#         plt.pause(.001)
#         ax.cla()
# =============================================================================
    
    return lattice





def randomly_infected(lattice,N_lat):

    x,y,z = randomly_choose_an_agent(lattice,N_lat)
    if (np.int(lattice[x,y,z]) == 1):
        lattice[x,y,z] = 3
    if (np.int(lattice[x,y,z]) == 5):
        lattice[x,y,z] = 7
    if (np.int(lattice[x,y,z]) == 9):
        lattice[x,y,z] = 11
    return lattice

def randomly_immune(lattice,N_lat):
    
    i,j,k = randomly_choose_an_agent(lattice,N_lat)

    if lattice[i,j,k] == 1:
        lattice[i,j,k] = 4
        
    if lattice[i,j,k] == 5:
        lattice[i,j,k] = 8
        
    if lattice[i,j,k] == 9:
        lattice[i,j,k] = 12
        

    return lattice


def randomly_choose_an_agent(lattice,N_lat):

    rca = np.argwhere((lattice != 0) & ( (lattice != 3) |  (lattice != 7) | (lattice != 11) ))
    size_len = len(rca)
    rand_ind = random.randint(0,size_len-1)
    x = rca[rand_ind,0]
    y = rca[rand_ind,1]
    z = rca[rand_ind,2]


    return x,y,z





def check_all_immunity(lattice, size):
    
    y_R = np.argwhere((lattice == 4))
    m_R = np.argwhere((lattice == 8))
    o_R = np.argwhere((lattice == 12))
    y = len(y_R)
    m = len(m_R)
    o = len(o_R)

    return y,m,o




def infection(lattice,N_lat,prob_par,x,Counta,red,tot_inf_per,day,N_time):
    
    exp_time = np.exp(-day/N_time)
    for i,j,k in zip(x[:,0],x[:,1],x[:,2]):
        S_awa_ind = np.argwhere((Counta[:,0] == i) & (Counta[:,1] == j) & (Counta[:,2] == k))
        if lattice[i,j,k] in [3,7,11]:
            if Counta[S_awa_ind,8] == 1:

                probY = exp_time*(prob_par[0]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probM = exp_time*(prob_par[1]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probO = exp_time*(prob_par[2]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                #probY = (prob_par[0]*(1 - red))
                #probM = (prob_par[1]*(1 - red))
                #probO = (prob_par[2]*(1 - red))
               
                lattice = nearest_neighbor_infection(i,j,k,lattice,N_lat,probY,probM,probO,N_lat)
            else:
                probY = exp_time*(prob_par[0]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probM = exp_time*(prob_par[1]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probO = exp_time*(prob_par[2]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                #probY = (prob_par[0]*(1))
                #probM = (prob_par[1]*(1))
                #probO = (prob_par[2]*(1))                
                
                lattice = nearest_neighbor_infection(i,j,k,lattice,N_lat,probY,probM,probO,N_lat)  
        elif lattice[i,j,k] in [2,6,10]:
            if Counta[S_awa_ind,8] == 1:
                probYSI = exp_time*(prob_par[5]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probMSI = exp_time*(prob_par[6]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probOSI = exp_time*(prob_par[7]*(1 - red))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                #probYSI = (prob_par[5]*(1 - red))
                #probMSI = (prob_par[6]*(1 - red))
                #probOSI = (prob_par[7]*(1 - red))    
                
                lattice = nearest_neighbor_infection(i,j,k,lattice,N_lat,probYSI,probMSI,probOSI,N_lat)
            else:
                probYSI = exp_time*(prob_par[5]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probMSI = exp_time*(prob_par[6]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                probOSI = exp_time*(prob_par[7]*(1))*(np.sin(2*mth.pi*(tot_inf_per))*tot_inf_per + (1.1))
                #probYSI = (prob_par[5]*(1))
                #probMSI = (prob_par[6]*(1))
                #probOSI = (prob_par[7]*(1)) 
                
                lattice = nearest_neighbor_infection(i,j,k,lattice,N_lat,probYSI,probMSI,probOSI,N_lat) 
    return lattice

def nearest_neighbor_infection(i,j,k,lattice,size,probY,probM,probO,N_lat):

    ii_ran = range(i-1,i+2)
    jj_ran = range(j-1,j+2)
    kk_ran = range(k-1,k+2)
    for ii in ii_ran:  
       iii = ii % N_lat
           
       for jj in jj_ran:           
           jjj = jj % N_lat
           for kk in kk_ran:                                                         
               kkk = kk % N_lat
               spot_check = np.int(lattice[iii,jjj,kkk])
               if ([iii, jjj, kkk] == [i, j, k]) or (spot_check not in [1,5,9]):
                   continue
               elif (spot_check  == 1):
                   S_p = probY
               elif (spot_check == 5):
                   S_p = probM
               else:
                   S_p = probO
               rand_val = random.random()
               #print(spot_check,S_p)
               if ((spot_check in [1,5,9])) and (S_p > rand_val):
                   lattice[iii,jjj,kkk] = lattice[iii,jjj,kkk]+1

               
    return lattice
def initial_counter(lattice,time_con, N_time,minf_par):
    Ts_min = np.int(minf_par[0])
    Ts_max = np.int(minf_par[1])
    Ti_min = np.int(minf_par[2])
    Ti_max = np.int(minf_par[3])
    size = np.int(minf_par[4])
    Ts_avg = np.int(minf_par[5])
    Ti_avg = np.int(minf_par[6])
    Ts_scale = np.int(minf_par[7])
    Ti_scale = np.int(minf_par[8])
    Tawa_min = np.int(minf_par[9])
    Tawa_max = np.int(minf_par[10])
    Tawa_avg = np.int(minf_par[11])
    Tawa_scale = np.int(minf_par[12])
    
    
    x,y,z = (lattice>0).nonzero()
    sample_i = random_sample(Ts_min,Ts_max,Ts_avg,Ts_scale,size)
    sample_r = random_sample(Ti_min,Ti_max,Ti_avg,Ti_scale,size)
    sample_aware = random_sample(Tawa_min,Tawa_max,Tawa_avg,Tawa_scale,size)
    sample_s = random_sample(5,15,10,15,size)
    for i in range(0,size):
        # Column 0-2 -> [i,j,k]
        time_con[i,0] = x[i] 
        time_con[i,1] = y[i] 
        time_con[i,2] = z[i] 
        # age at i,j,k
        time_con[i,3] = lattice[x[i],y[i],z[i]]
        # Columns 4 & 5 are counters
        time_con[i,4] = 0
        time_con[i,5] = 0
        # Columns 6 & 7 are time periods for SI to I and I to R
        time_con[i,6] = sample_i[i]
        time_con[i,7] = sample_r[i]  
        # Keep track whose aware
        time_con[i,8] = 0
        # Counter For awareness
        time_con[i,9] = 0
        # time period of awareness
        time_con[i,10] = sample_aware[i]
        # Counter for R to S
        time_con[i,11] = 0
        time_con[i,12] = sample_s[i]
    return time_con




def change_status(lattice,time_count,total_pop):
    
    
    find_SI_c = np.argwhere((time_count[:,3] == 2) | (time_count[:,3] == 6) | (time_count[:,3] == 10))
    
    for lm in find_SI_c:
        time_count[lm,4] += 1
        if np.int(time_count[lm,4]) == np.int(time_count[lm,6]):
            time_count[lm,3] += 1
            lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] += 1
            time_count[lm,4] = 0
          
    
    find_I_c = np.argwhere((time_count[:,3] == 3) | (time_count[:,3] == 7) | (time_count[:,3] == 11))
    
    for lm in find_I_c:
        time_count[lm,5] += 1
        if np.int(time_count[lm,5]) == np.int(time_count[lm,7]):
            time_count[lm,3] = time_count[lm,3] + 1
            lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] = lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] + 1    
            time_count[lm,5] = 0
            
    find_Awa_c = np.argwhere((time_count[:,8] == 1) )
    
    for lm in find_Awa_c:
        time_count[lm,9] += 1
        if np.int(time_count[lm,9]) == np.int(time_count[lm,10]):
            time_count[lm,8] = 0
            time_count[lm,9] = 0
    
# =============================================================================
    find_R_c = np.argwhere((time_count[:,3] == 4) | (time_count[:,3] == 8) | (time_count[:,3] == 12))
     
    for lm in find_R_c:
        time_count[lm,11] += 1
        if np.int(time_count[lm,11]) == np.int(time_count[lm,12]):
            if time_count[lm,3] == 4:
                time_count[lm,3] = 1
                lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] = 1 
                time_count[lm,11] = 0
                 
            elif time_count[lm,3] == 8:
                time_count[lm,3] = 5
                lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] = 5 
                time_count[lm,11] = 0
                 
            elif time_count[lm,3] == 12: 
                 
                time_count[lm,3] = 9
                lattice[np.int(time_count[lm,0]),np.int(time_count[lm,1]),np.int(time_count[lm,2])] = 9  
                time_count[lm,11] = 0
                 
# =============================================================================
    return lattice, time_count

def awareness_function(num_infected_perc,delta):
    
    awared = total_pop*(1.0-np.exp(-delta*(num_infected_perc)))
    return awared
def infection_spreading(lattice,N_lat,step_pop,N_time,prob_par,info,n_pop,Counta,beta_vec,minf_par):
    # Define parameters
    total_pop = minf_par[4]
    pop_ran =  range(0,step_pop)
    b_y = beta_vec[0]
    b_m = beta_vec[1]
    b_o = beta_vec[2]
    delta = beta_vec[3]
    red = beta_vec[4]
    h = beta_vec[5]
    # Start of infection model is Feb 9th: First act of social distancing starts on March 22
    law_1 = np.int(42)
    law_2 = np.int(law_1+16)
    lifted = np.int(law_2+23)
    
    # Randomly infect one person
    lattice = randomly_infected(lattice,N_lat)
    #randonly choose 2.5% immune
    initial_immune = False
    while (not initial_immune):
        lattice = randomly_immune(lattice,N_lat)
        y,m,o = check_all_immunity(lattice,N_lat)
        if (y + m + o) == np.int(0.025*total_pop):
            initial_immune = True
    Counta = initial_counter(lattice,Counta, N_time,minf_par)
    
    # Time evolve
    for tt in range(0,N_time+1):
        # Infect people at start of a new day
        #lattice = infection(lattice,N_lat,prob_par)
        #Counta = change_S_SI(lattice, Counta,N_lat,total_pop)       
        # Find out how many is infected
        yxS, yyS,yzS = (lattice == 1).nonzero()
        mxS, myS,mzS = (lattice == 5).nonzero()
        oxS, oyS,ozS = (lattice == 9).nonzero()
        yS = len(yxS)
        mS = len(mxS)
        oS = len(oxS)
        
        yxSI, yySI,yzSI = (lattice == 2).nonzero()
        mxSI, mySI,mzSI = (lattice == 6).nonzero()
        oxSI, oySI,ozSI = (lattice == 10).nonzero()
        ySI = len(yxSI)
        mSI = len(mxSI)
        oSI =len(oxSI)
        
        yxI, yyI,yzI = (lattice == 3).nonzero()
        mxI, myI,mzI = (lattice == 7).nonzero()
        oxI, oyI,ozI = (lattice == 11).nonzero()
        yI = len(yxI)
        mI = len(mxI)
        oI = len(oxI)
        
        yxR, yyR,yzR = (lattice == 4).nonzero()
        mxR, myM,mzR = (lattice == 8).nonzero()
        oxR, oyR,ozR = (lattice == 12).nonzero()
        yR = len(yxR)
        mR = len(mxR)
        oR = len(oxR)
        
            
        # Total S,SI,I,R
        Total_S = yS + mS + oS
        Total_SI = ySI + mSI + oSI
        Total_I = yI + mI + oI
        Total_R = yR + mR + oR
        
        # Move moveable agents        
        info[tt,0] = Total_S
        info[tt,1] = Total_SI 
        info[tt,2] = Total_I
        info[tt,3] = Total_R
        
        info[tt,4] = yS
        info[tt,5] = ySI
        info[tt,6] = yI
        info[tt,7] = yR  
        
        info[tt,8] = mS
        info[tt,9] = mSI
        info[tt,10] = mI
        info[tt,11] = mR

        info[tt,12] = oS
        info[tt,13] = oSI
        info[tt,14] = oI
        info[tt,15] = oR
        info[tt,16] = tt          
        
        tot_inf_per = Total_I/total_pop
        
        num_of_awared = np.round(awareness_function(tot_inf_per, delta))    
        
        Naware_ind = np.argwhere((Counta[:,8] != 1))
        for an in range(len(Naware_ind)):
            if an > num_of_awared-1:
                break
            else:
                samp = random.sample(range(len(Naware_ind)),k = 1)
                anC = [Naware_ind[iv] for iv in samp]
                if  Counta[anC,8] < 1:                    
                    Counta[anC,8] = 1
        
        inf_arr = np.argwhere((lattice == 3) | (lattice == 7) | (lattice == 11) |(lattice == 2) |(lattice == 6) |(lattice == 10))
        
        lattice = infection(lattice,N_lat,prob_par,inf_arr,Counta,red,tot_inf_per,tt,N_time)
       
        lat_newi_ind = np.argwhere((lattice == 2) | (lattice == 6) | (lattice == 10) )

        for vl in range(len(lat_newi_ind)):
            si_ind_c = np.argwhere((Counta[:,0] == lat_newi_ind[vl,0]) & (Counta[:,1] == lat_newi_ind[vl,1]) & (Counta[:,2] == lat_newi_ind[vl,2]))
            if Counta[si_ind_c[0,0],3] in [1,5,9]:
                Counta[si_ind_c[0,0],3] = Counta[si_ind_c[0,0],3]+1  
        
        if tt == law_1:
            h = 1.4
        elif tt == law_2:
            h = 1.6
        elif tt == lifted:
            h = 1.2

        for pop_step in pop_ran:



            i,j,k =  randomly_choose_an_agent(lattice, N_lat)
            #print(lattice[i,j,k])
            if lattice[i,j,k] == 0:
                continue 

            
            if (lattice[i,j,k] in [3,7,11]):
                continue
            
            
            
            m,n,o = random_movement(i,j,k,N_lat)
            
            
            if lattice[m,n,o] > 0:
                continue 
            
            # Find ind of i,j,k in coutner to get info on awareness
            c_ai = np.argwhere((Counta[:,0] == i) & (Counta[:,1] == j) & (Counta[:,2] == k))

            aware_move2 = Counta[c_ai,8]
            aware_move1 = aware_move2[0]
            aware_move = aware_move1[0]
            age1 = lattice[i,j,k]
            if (1 <= age1 <= 4):
                age = [1,2,4]
                beta = b_y + red**(aware_move)
            elif (5 <= age1 <= 8):
                age = [5,6,8]
                beta = b_m + red**(aware_move)
            else:
                age = [9,10,12]
                beta = b_o + red**(aware_move)
                
                
            N_1, is_adjacent_ijk  = nnPBC_sum(i, j, k, lattice, i,j,k,N_lat,beta,age) 
    
            N_2, is_adjacent_mno  = nnPBC_sum(m,n,o,lattice,i,j,k, N_lat,beta,age) 

            if (is_adjacent_mno == True) and (lattice[i,j,k] not in [4,8,12]):
                continue
            else: 
                #print('am I working', is_I)
                
                #print(aware_move)
                dE = (N_2 - N_1) + h*aware_move*(N_1 - N_2)
                
                prob_n = np.exp(dE)
                rand_val =  random.random()
                #print(rand_val,prob_n,aware_move)
                if  rand_val < prob_n:
                    #print('am I working', lattice[i,j,k])
                    # moved with probability e^(beta(delatN))
                    lattice[m,n,o] = lattice[i,j,k] 
                    lattice[i,j,k] = 0                 
                    #lattice = infection(lattice,N_lat,prob_par)
                    #Counta = change_S_SI(lattice, Counta,N_lat,total_pop)
                    c_ind = np.argwhere((Counta[:,0] == i) & (Counta[:,1] == j) & (Counta[:,2] == k))
                    Counta[c_ind,0:3] = [m,n,o]
                    
            
                #lattice = infection(lattice,N_lat,prob_par)
                #Counta = change_S_SI(lattice, Counta,N_lat,total_pop)   
# =============================================================================
#                     xany,yany,zany = (lattice==1).nonzero()
#                     x,y,z = (lattice ==  1).nonzero()
#                     x2,y2,z2 = (lattice == 2).nonzero()
#                     x3,y3,z3 = (lattice == 3).nonzero()
#                     x4,y4,z4 = (lattice == 4).nonzero()
#                     xm,ym,zm = (lattice ==  5).nonzero()
#                     xm2,ym2,zm2 = (lattice == 6).nonzero()
#                     xm3,ym3,zm3 = (lattice == 7).nonzero()
#                     xm4,ym4,zm4 = (lattice == 8).nonzero()
#                     xo,yo,zo = (lattice ==  9).nonzero()
#                     xo2,yo2,zo2 = (lattice == 10).nonzero()
#                     xo3,yo3,zo3 = (lattice == 11).nonzero()
#                     xo4,yo4,zo4 = (lattice == 12).nonzero()
#                 
#                     ax.scatter(x,y,z, zdir='z',c='Green',marker = 'o', alpha = 0.5)
#                     ax.scatter(x2,y2,z2, zdir='z',c='blue', marker = 'o', alpha = 0.5)
#                     ax.scatter(x3,y3,z3, zdir='z',c='red',marker ='o', alpha = 0.5)
#                     ax.scatter(x4,y4,z4, zdir='z',c='black',marker ='o', alpha = 0.5)
#         
#                     ax.scatter(xm,ym,zm, zdir='z',c='Green',marker = 'o', alpha = 0.75)
#                     ax.scatter(xm2,ym2,zm2, zdir='z',c='Blue', marker = 'o', alpha = 0.75)
#                     ax.scatter(xm3,ym3,zm3, zdir='z',c='red',marker ='o', alpha = 0.75)
#                     ax.scatter(xm4,ym4,zm4, zdir='z',c='black',marker ='o', alpha = 0.75)
#                 
#                     ax.scatter(xo,yo,zo, zdir='z',c='green',marker = 'o')
#                     ax.scatter(xo2,yo2,zo2, zdir='z',c='blue', marker = 'o')
#                     ax.scatter(xo3,yo3,zo3, zdir='z',c='red',marker ='o')
#                     ax.scatter(xo4,yo4,zo4, zdir='z',c='black',marker ='o')
#                 
#                     plt.draw()
#                     plt.pause(.001)
#                     ax.cla()
# =============================================================================
                    
        
        #lattice = infection(lattice,N_lat,prob_par)
        #Counta = change_S_SI(lattice, Counta,N_lat,total_pop)   
        lattice, Counta = change_status(lattice,Counta,total_pop)

# =============================================================================
#         if tt == np.int(N_time/4):
#             print("25%")
#         if tt == np.int(N_time/2):
#             print("50%")
#         if tt == np.int(3*N_time/4):
#             print("75%")            
# =============================================================================
    return lattice,info,Counta


def avg_spreading(info_avg,N_avg,N_lat,step_pop,N_time,prob_par,information,n_pop,beta_vec,N_trans,minf_par):
       
    for oo in range(0,N_avg):
        start_Int = time.time()
        lattice, Counta = initialize_model(N_lat,beta_vec,N_trans,n_pop)
        finish_Int = time.time()
        print('Finished initializing',oo,  'in',finish_Int-start_Int, 'secs')
        start_cal = time.time()
        lattice,information,Counta = infection_spreading(lattice, N_lat, step_pop, N_time, prob_par, information, n_pop, Counta, beta_vec,minf_par)
        finish_cal = time.time()
        print('Finished calculating',oo,' in',finish_cal-start_cal, 'secs','Prediction to end in:',N_avg*finish_Int-(oo+1)*finish_Int,'s')       
        info_avg[:,0,oo] == information[:,0]
        info_avg[:,1,oo] == information[:,1]
        info_avg[:,2,oo] == information[:,2]
        info_avg[:,3,oo] == information[:,3]
        info_avg[:,4,oo] == information[:,4]
        info_avg[:,5,oo] == information[:,5]
        info_avg[:,6,oo] == information[:,6]
        info_avg[:,7,oo] == information[:,7]
        info_avg[:,8,oo] == information[:,8]
        info_avg[:,9,oo] == information[:,9]
        info_avg[:,10,oo] == information[:,10]
        info_avg[:,11,oo] == information[:,11]
        info_avg[:,12,oo] == information[:,12]
        info_avg[:,13,oo] == information[:,13]
        info_avg[:,14,oo] == information[:,14]
        info_avg[:,15,oo] == information[:,15]
        info_avg[:,16,oo] == information[:,16]

    for nn in range(N_time):
        information[nn,0] == np.mean(info_avg[nn,0,:])
        information[nn,1] == np.mean(info_avg[nn,1,:])
        information[nn,2] == np.mean(info_avg[nn,2,:])
        information[nn,3] == np.mean(info_avg[nn,3,:])
        information[nn,4] == np.mean(info_avg[nn,4,:])
        information[nn,5] == np.mean(info_avg[nn,5,:])
        information[nn,6] == np.mean(info_avg[nn,6,:])
        information[nn,7] == np.mean(info_avg[nn,7,:])
        information[nn,8] == np.mean(info_avg[nn,8,:])
        information[nn,9] == np.mean(info_avg[nn,9,:])
        information[nn,10] == np.mean(info_avg[nn,10,:])
        information[nn,11] == np.mean(info_avg[nn,11,:])
        information[nn,12] == np.mean(info_avg[nn,12,:])
        information[nn,13] == np.mean(info_avg[nn,13,:])
        information[nn,14] == np.mean(info_avg[nn,14,:])
        information[nn,15] == np.mean(info_avg[nn,15,:])
        information[nn,16] == np.mean(info_avg[nn,16,:])
    #draw_fig(lattice,N_lat)
    return information, Counta

def initialize_model(N_lat, beta_vec, N_trans,n_pop):
    lattice = np.zeros((N_lat,N_lat,N_lat))
    total_pop = np.sum(n_pop)
    Counta = np.zeros((np.int(total_pop),13))
    lattice = fill_lattice(lattice,n_pop,N_lat)
    lattice = Population_equilibrate(lattice, N_lat, beta_vec, N_trans)
    
    return lattice, Counta


def draw_fig(lattice,N_lat):
    #ax = fig.add_subplot(111,projection='3d')
    x,y,z = (lattice ==  1).nonzero()
    x2,y2,z2 = (lattice == 2).nonzero()
    x3,y3,z3 = (lattice == 3).nonzero()
    x4,y4,z4 = (lattice == 4).nonzero()
    xm,ym,zm = (lattice ==  5).nonzero()
    xm2,ym2,zm2 = (lattice == 6).nonzero()
    xm3,ym3,zm3 = (lattice == 7).nonzero()
    xm4,ym4,zm4 = (lattice == 8).nonzero()
    xo,yo,zo = (lattice ==  9).nonzero()
    xo2,yo2,zo2 = (lattice == 10).nonzero()
    xo3,yo3,zo3 = (lattice == 11).nonzero()
    xo4,yo4,zo4 = (lattice == 12).nonzero()

    ax.scatter(x,y,z, zdir='z',c='Green',marker = 'o', alpha = 1)
    ax.scatter(x2,y2,z2, zdir='z',c='blue', marker = 'o', alpha = 1)
    ax.scatter(x3,y3,z3, zdir='z',c='red',marker ='o', alpha = 1)
    ax.scatter(x4,y4,z4, zdir='z',c='black',marker ='o', alpha = 1)
    
    ax.scatter(xm,ym,zm, zdir='z',c='Green',marker = 'o', alpha = 1)
    ax.scatter(xm2,ym2,zm2, zdir='z',c='Blue', marker = 'o', alpha = 1)
    ax.scatter(xm3,ym3,zm3, zdir='z',c='red',marker ='o', alpha = 1)
    ax.scatter(xm4,ym4,zm4, zdir='z',c='black',marker ='o', alpha = 1)
    
    ax.scatter(xo,yo,zo, zdir='z',c='green',marker = 'o',alpha = 1)
    ax.scatter(xo2,yo2,zo2, zdir='z',c='blue', marker = 'o',alpha = 1)
    ax.scatter(xo3,yo3,zo3, zdir='z',c='red',marker ='o',alpha = 1)
    ax.scatter(xo4,yo4,zo4, zdir='z',c='black',marker ='o',alpha = 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(0,N_lat)
    ax.set_ylim3d(0,N_lat)
    ax.set_zlim3d(0,N_lat)
    return
def random_sample(t_min,t_max,t_mean,t_scale,t_size):
    

    size_c = np.int(t_size)
    avg = np.float(t_mean)
    samp = np.arange(t_min,t_max)
    prob = ss.norm.pdf(samp,loc = avg, scale = t_scale)
    prob = prob/prob.sum()
    sample = np.random.choice(samp, size = size_c, p = prob)
    return sample


# =============================================================================
# 
# MAIN CODE
# 
# =============================================================================
    


#N_lat,n_pop,beta,N_trans = parameters(10,60,90,50,1,30000)
print("Please input Size of lattice:")
N_lat = 12#int(input()) 
print(N_lat)
print("Please input total population:")
N_tot_pop =500#int(input()) 
print(N_tot_pop)
print("Please input total time integrated:")
N_time = 300#int(input()) 
print(N_time)  
N_young_pop = np.int(0.3*N_tot_pop)
N_middle_pop = np.int(0.45*N_tot_pop)
N_old_pop = np.int(0.25*N_tot_pop)
#lat,Y,M,O,beta_y,beta_m,beta_o,delta,red,h,trans #np.int(N_tot_pop*4)
N_lat,n_pop,beta_vec,N_trans = parameters(N_lat,N_young_pop,N_middle_pop,N_old_pop,0.6,0.8,1,1.5,0.075,1.2,150000)
print("Please input number of observations:")
N_avg = 1#int(input()) 
print(N_avg)


total_pop = np.sum(n_pop)

Ts_min = 4; Ts_max = 12;Ti_min=4;Ti_max=12;Ts_avg=8;Ti_avg=8;Ts_scale=Ts_avg/3;Ti_scale=Ti_avg/3
Tawa_min = 1;Tawa_max=12;Tawa_avg=6;Tawa_scale = Tawa_avg/3

parameters_inf = np.zeros((8,))

parameters_inf,N_time,pop_step = parameters_infection(0.02,0.1,0.11,7,7,.02,.1,.11,N_time, np.int(N_tot_pop*1))
inform_avg = np.zeros((N_time+1,17,N_avg))



minf_par = more_parameters(Ts_min,Ts_max,Ti_min,Ti_max,total_pop,Ts_avg,Ti_avg,Ts_scale,Ti_scale, \
Tawa_min,Tawa_max,Tawa_avg,Tawa_scale)

information = np.zeros((N_time+1,17))
Counta = np.zeros((np.int(total_pop),13))

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111,projection='3d')
information, Counta = avg_spreading(inform_avg,N_avg,N_lat,pop_step,N_time,parameters_inf,information,n_pop,beta_vec,N_trans,minf_par)

#plt.interactive(True)
# =============================================================================
fig = plt.figure(figsize=(15, 15))

sp =  fig.add_subplot(2, 2, 1 )
plt.title('Avg Susceptible Infected per day:' +str(np.mean(information[:,0])),fontsize = 15)
#plt.plot(information[:,16],information[:,0]/total_pop,'green', linestyle = "solid",label= "S")
plt.bar(information[:,16],(information[:,0])/N_tot_pop,color = 'green', label = "Susceptible")
#plt.plot(information[:,16],information[:,2]/total_pop,'red', linestyle = "solid",label = "I")
#plt.bar(information[:,16],information[:,3]/total_pop,color = 'black', linestyle  = "solid", label = "R")
plt.legend(loc='upper right')
plt.vlines(42, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16+23, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.xlabel('Time, (days)')
plt.ylabel("Percentage of Population, %")
plt.ylim(0,1)
#plt.ylim(0,1)
plt.xlim(0,N_time)
plt.show()

# =============================================================================
sp =  fig.add_subplot(2, 2,3 )
plt.title('Avg SI Infected per day:' +str(np.mean(information[:,1])),fontsize = 15)
#plt.plot(information[:,16],information[:,0]/total_pop,'green', linestyle = "solid",label= "S")
plt.bar(information[:,16],(information[:,1])/N_tot_pop,color = 'blue', label = "No Symptoms")
#plt.plot(information[:,16],information[:,2]/total_pop,'red', linestyle = "solid",label = "I")
plt.vlines(42, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16+23, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.legend(loc='upper right')
plt.xlabel('Time, (days)')
plt.ylabel("Percentage of Population, %")
#plt.ylim(0,.1)
#plt.ylim(0,1)
plt.xlim(0,N_time)
plt.show()

sp =  fig.add_subplot(2, 2,4 )
plt.title('Avg Confirmed Infected per day:' +str(np.mean(information[:,2])),fontsize = 15)
#plt.plot(information[:,16],information[:,0]/total_pop,'green', linestyle = "solid",label= "S")
plt.bar(information[:,16],(information[:,2])/N_tot_pop,color = 'red', label = "Infected")
plt.vlines(42, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16+23, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.legend(loc='upper right')
plt.xlabel('Time, (days)')
plt.ylabel("Percentage of Population, %")
#plt.ylim(0,.1)
#plt.ylim(0,1)
plt.xlim(0,N_time)
plt.show()

sp =  fig.add_subplot(2, 2, 2 )
plt.title('Avg Recovered per day:' +str(np.mean(information[:,3])),fontsize = 15)
#plt.plot(information[:,16],information[:,0]/total_pop,'green', linestyle = "solid",label= "S")
plt.bar(information[:,16],(information[:,3])/N_tot_pop,color = 'black', label = "Recovered")
plt.vlines(42, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.vlines(42+16+23, 0,max((information[:,1])/N_tot_pop),linestyles = 'dotted')
plt.legend(loc='upper right')
plt.xlabel('Time, (days)')
plt.ylabel("Percentage of Population, %")
plt.ylim(0,1)
#plt.ylim(0,1)
plt.xlim(0,N_time)
plt.show()

fig.savefig("COVID_Total.pdf", format = 'pdf', bbox_inches = "tight")