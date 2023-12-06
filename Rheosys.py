import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from random import random


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...
def sine (period):
    return np.sin(period)

def cos (period):
    return np.cos(period)

def DB (arr):
    ref = 1
    decibel=[]
    for i in arr:
        if i!=0:
            decibel.append(20 * np.log10(abs(i) / ref))
        else:
            decibel.append(-60)

    return decibel

def rms(arr):
    rms = np.sqrt(np.mean(np.square(arr)))
    return rms

def crest_fac(frequency):
    return (max(abs(frequency)))/rms(frequency)

def phase_lin(N,f_s,Tau,t):
    return -Tau*2*np.pi*(t/N)*f_s

def phase_rand():
    return np.pi*random()

def phase_schroeder():
    return