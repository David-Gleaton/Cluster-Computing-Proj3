from mpi4py import MPI
import numpy as np
"""
David Gleaton
Bucket1.c rewritten to pyBucket1.py
IMPORTANT
    -This was written using a pytorch env, with conda packages
"""
#----------Phase 1----------
#define the number of rand ints
N = 64

#initialize comm and rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

#create rawNum and sortNum arrays
rawNum = np.zeros(N, dtype="int")

#Populate rawNum with ints
if(rank == root):
    rawNum = np.random.randint(0,N, size=(N))

#bcast the random ints
rawNum = comm.bcast(rawNum, root=0)


#----------Phase 2----------
#Each process only works with numbers within their assigned interval
counter = 0
local_min = rank * (N/size)
local_max = (rank+1) * (N/size)
for i in range(0, N):
    if((rawNum[i] >= local_min) and (rawNum[i] < local_max)):
        counter += 1
comm.Barrier()
print("For rank ", rank, " max is ", local_max, ", min is ", local_min, ", and there are ", counter, " elements in rawNum that falls within max and min \n")




#Each process creates its own bucket containing values that fall within its iterval
local_bucket = np.zeros(counter, dtype="int")
counter = 0
for i in range(0, N):
    if((rawNum[i] >= local_min) and (rawNum[i] < local_max)):
        local_bucket[counter] = rawNum[i] 
        counter += 1


#Insertion sort
for i in range(0, counter):
    for j in range(0, counter):
        if(local_bucket[i] < local_bucket[j]):
            tmp = local_bucket[i]
            local_bucket[i] = local_bucket[j]
            local_bucket[j] = tmp
comm.Barrier()
for i in range(0, counter):
    print(rank, local_bucket[i], "\n")


#----------Phase 3----------
#Honestly, I abandoned the disp idea and moved towards this implementation
#Somehow, and I don't know why, Gatherv only wants recvbuf and not sortNum as its name
#so I caved, and recvbuf is where the sorted arrays are collected

#Create np.array for gather later
sendbuf = np.array(local_bucket)

#Get local_bucket sizes
sendcounts = np.array(comm.gather(len(sendbuf), root))

#Create recvbuf for gathering the arrays
if rank == root:
    recvbuf = np.empty(sum(sendcounts), dtype=int)
else:
    recvbuf = None

#Gatherv call to collect the arrays
comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
comm.Barrier()
#Print out the final collection
if(rank == root):

    print("Before sort: \n" )
    print(rawNum)
    print("\nAfter sort: \n")
    print("{}".format(recvbuf))       



