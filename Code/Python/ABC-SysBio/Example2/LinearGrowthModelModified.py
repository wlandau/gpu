from math import *
from numpy import *
from abcsysbio.relations import *

def modelfunction((species1,)=(2.0,),dt=0,parameter=(1.0,10.0,1.0,),time=0):

	compartment1=parameter[0]
	parameter1=parameter[1]
	parameter2=parameter[2]


	d_species1=((1.0)*(parameter1*compartment1)+(-1.0)*(parameter2*species1)+0)/compartment1

	noise_species1=(0.1)*sqrt(parameter1*compartment1)*random.normal(0.0,sqrt(dt),1) + (0.1)*sqrt(parameter2*species1)*random.normal(0.0,sqrt(dt),1)[0] + 0

	return((d_species1,),(noise_species1, ))


def rules((species1,),(compartment1,parameter1,parameter2,),t):



	return((species1,),(compartment1,parameter1,parameter2,))

