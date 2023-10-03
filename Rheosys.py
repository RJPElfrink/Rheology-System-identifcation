import numpy as np
import scipy as sc


class excitation():
    #Excitation signals used as input for the analysis,
    #   input must be defined as one value with the unit of ...
    #   output is defined with the unit of ...

    def sin (x):
        ouput=np.sin(x)
        return ouput

    def cos (x):
        ouput=np.sin(x)
        return ouput


class ode():
    #ODE included in rheological models that include mass-spring damper systems to describe the material properties
    #input variables needed for these solvers ... with units ...
    #output variables given from these solvers ... with units ...

    def maxwell():

    def kelvin():

