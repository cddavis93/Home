from mpi4py import MPI
import numpy as np
import math
import sys
import cmath

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = MPI.Wtime()

real = np.array([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
imag = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

#real = np.array([0.,1.,2.,3.,4.,5.,6.,7.])
#imag = np.array([8.,9.,10.,11.,12.,13.,14.,15.])

assert len(real) == len(imag)
n = len(real)

outreal = np.zeros(n)
outimag = np.zeros(n)
loc_outreal = np.zeros(n/size)
loc_outimag = np.zeros(n/size)

loc_real = np.zeros(n/size)
loc_imag = np.zeros(n/size)

mas_sumreal = 0.0
mas_sumimag = 0.0

#comm.Scatter(real,loc_real,root=0)
#comm.Scatter(imag,loc_imag,root=0)
comm.Bcast(real,root=0)
comm.Bcast(imag,root=0)

#print (rank, loc_real, loc_imag)


for k in range(rank*n/size,n/size*(rank+1)): #n/size,size):  # For each output element
	loc_sumreal = 0.0
	loc_sumimag = 0.0

	#print("loop")
	#print(k)

	for t in range(0,n): #/size): #rank*n/size,n/size*(rank+1)): #2*rank,2*(rank+1)): #0,n/size,size):  # For each input element
		angle = 2 * math.pi * (t+ rank*n/size) * k / n #t * k / n # 
		loc_sumreal +=  real[t] * math.cos(angle) + imag[t] * math.sin(angle)
		loc_sumimag += -real[t] * math.sin(angle) + imag[t] * math.cos(angle)

	if (k == rank*n/size):
		loc_outreal = np.array([loc_sumreal])
		loc_outimag = np.array([loc_sumimag])
	else:	
		loc_outreal = np.append(loc_outreal,loc_sumreal)
		loc_outimag = np.append(loc_outimag,loc_sumimag)
	
	#print (rank, k, loc_outreal, loc_outimag)
	#print (loc_outreal)
	#print (loc_outimag)


comm.Gather(loc_outreal,outreal,root=0)
comm.Gather(loc_outimag,outimag,root=0)

if (rank == 0):
	finish = MPI.Wtime()
	time = finish - start
	print ("Time: ")
	print time
	print ("Final arrays")	
	print outreal
	print outimag

