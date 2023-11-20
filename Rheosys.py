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




import math
import sys

def find_lcm(f_0, end_range):
    lcm = f_0
    for i in range(2, end_range + 1):
        lcm = math.lcm(lcm, f_0 * i)
    return lcm

def is_divisible(N, f_0, i):
    # Check if N divided by (f_0 * i) has no remainder
    return N % (f_0 * i) == 0

def window(f_0, N, end_range):

    recommended_N_larger = N
    recommended_N_smaller = N


    # Search for the closest suitable N in both directions
    while not (any(is_divisible(recommended_N_larger, f_0, i) for i in range(1, end_range + 1)) or any(is_divisible(recommended_N_smaller, f_0, i) for i in range(1, end_range + 1))):
        recommended_N_larger += 1
        recommended_N_smaller -= 1

    # Choose the closest suitable N
    if any(is_divisible(recommended_N_larger, f_0, i) for i in range(1, end_range + 1)) and any(is_divisible(recommended_N_smaller, f_0, i) for i in range(1, end_range + 1)):
        recommended_N = recommended_N_larger if abs(N - recommended_N_larger) < abs(N - recommended_N_smaller) else recommended_N_smaller
    elif any(is_divisible(recommended_N_larger, f_0, i) for i in range(1, end_range + 1)):
        recommended_N = recommended_N_larger
    else:
        recommended_N = recommended_N_smaller



    # Exit the program if N is not divisible by f_0
    if not any(is_divisible(N, f_0, i) for i in range(1, end_range + 1)):
        x=f_0*recommended_N
        print(f"Selected N is not divisible by f_0.")
        print(f"The recommended N for divisibility by ({f_0} * i) for i in the range 1 to {end_range} around {N} is {recommended_N}. To avoid leakage the sampling frequency is advised to be {x} [Hz]")

