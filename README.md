# KmeansMenny

Implementation of the kmeans c algorithm in a parallel and efficient way.

A little about kmeans - the algorithm receives large amounts of information
which it should categorize the raw information is displayed as points on a suitable graph
and the algorithm groups k groups smartly even if the information changes in time and regardless
of the number of characteristics the information holds

Through mpi the code runs on 3 computers when one computer is used as a master and besides 
performing computational operations himself it also sends and receives data from two other computers (slaves).

Using omp, each computer uses a large number of threads to perform the calculations required for the algorithm

And through cuda, the processing power of the GPU is used.

in this project the data read from the file input.txt
every row is a point 
all the points has 4 values
x, y represent the information values.
Whereas deltaX and deltaY represent the change in the info that occurred in time

the output.txt file created in the end of the program and here example 
error = 0.000050 operation time = 278.210669
Centers of the clusters :
cluster: 0:	 (0.000135	-0.000168)
cluster: 1:	 (0.000714	99.999352)
cluster: 2:	 (100.000359	100.000839)
cluster: 3:	 (119.999611	0.000381)


