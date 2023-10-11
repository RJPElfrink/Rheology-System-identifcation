import numpy as np
import scipy as sc


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...

def sin (time):
    ouput=np.sin(time)
    return ouput

def cos (x):
    ouput=np.cos(x)
    return ouput
