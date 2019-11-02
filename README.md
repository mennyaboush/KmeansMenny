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
