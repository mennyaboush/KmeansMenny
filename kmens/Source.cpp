#pragma once
#include "HeaderForC.h"

int main(int argc, char *argv[])
{
	int flag = 1, firstRound = 1;// 0 - need to recluster  , 1 - else
	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int numberOfPoints, k, limitIter;
	double qm, dt, maxTime;
	Point *points = NULL;
	ClusterPoint * clusters = NULL;
	int partSize;// = numberOfPoints / numprocs;
	int details[3];
	cudaError_t cudaStatus;

#pragma region MPI_init

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Get_processor_name(processor_name, &namelen);

	MPI_Status status;
#pragma endregion

#pragma region createDataType
	MPI_Datatype PointMPIType;
	MPI_Datatype type[STRUCT_SIZE_POINT]
		= { MPI_FLOAT,MPI_FLOAT,MPI_FLOAT ,MPI_FLOAT ,MPI_INT };
	int blocklen[STRUCT_SIZE_POINT] = { 1, 1, 1 ,1 ,1 };
	MPI_Aint disp[STRUCT_SIZE_POINT];

	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.speedX - (char *)&point;
	disp[3] = (char *)&point.speedY - (char *)&point;
	disp[4] = (char *)&point.clusterId - (char *)&point;

	MPI_Type_create_struct(STRUCT_SIZE_POINT, blocklen, disp,
		type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);

	MPI_Datatype ClusterPointMPIType;
	MPI_Datatype type2[STRUCT_SIZE_CLUSTER_POINT]
		= { MPI_FLOAT,MPI_FLOAT,MPI_INT ,MPI_FLOAT , MPI_FLOAT };
	int blocklen2[STRUCT_SIZE_CLUSTER_POINT] = { 1, 1, 1 ,1, 1 };
	MPI_Aint disp2[STRUCT_SIZE_POINT];
	disp2[0] = (char *)&clusterPoint.x - (char *)&clusterPoint;
	disp2[1] = (char *)&clusterPoint.y - (char *)&clusterPoint;
	disp2[2] = (char *)&clusterPoint.numbersOfPoints - (char *)&clusterPoint;
	disp2[3] = (char *)&clusterPoint.sumX - (char *)&clusterPoint;
	disp2[4] = (char *)&clusterPoint.sumY - (char *)&clusterPoint;
	MPI_Type_create_struct(STRUCT_SIZE_POINT, blocklen2, disp2,
		type2, &ClusterPointMPIType);
	MPI_Type_commit(&ClusterPointMPIType);

#pragma endregion

	if (myid == 0)
	{
		int i, iter;
		double error, time;
		double t1, t2;
		readFromFile(&points, &numberOfPoints, &k, &maxTime, &dt, &limitIter, &qm);
		partSize = numberOfPoints / numprocs;
		details[0] = partSize;
		details[1] = k;
		details[2] = numberOfPoints;
		t1 = MPI_Wtime();
		for (i = 1; i < numprocs; i++)
			MPI_Send(&details, 3, MPI_INT, i, 0, MPI_COMM_WORLD);

		clusters = creatClusters(k, points);
	
		/*div the points and send parts to slaves
		the master make cluster for his part and get the clusterd points from the slaves
		then the master marge the points
		but the bace location must to change after the marge! */

		error = qm + 1;
		for (iter = 0, time = 0; iter < limitIter && time <= maxTime && error > qm; iter++, time += dt)
		{
			printf("\nSTART iter %d\n", iter);
			fflush(stdout);
			if (time != 0)
			{
				cudaStatus = movePointsWithCuda(numberOfPoints, points, dt);
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "C-file cudaDeviceReset failed!");
					return 1;
				}
				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "C-file cudaDeviceReset failed!");
					return 1;
				}
			}

			do {

				int j;

				resetClusters(clusters, k);//can be write befor or after the "clusterPoints" method.

				for (i = 0; i < numprocs - 1; i++)
				{
					MPI_Send(clusters, k, ClusterPointMPIType, i + 1, 0, MPI_COMM_WORLD);
					MPI_Send(points + i*partSize, partSize, PointMPIType, i + 1, 0, MPI_COMM_WORLD);
				}


				clusterPoints(clusters, k, points + (numprocs - 1)*partSize, partSize + (numberOfPoints % numprocs), myid, firstRound);

				ClusterPoint* tempCluster = (ClusterPoint*)calloc(k, sizeof(ClusterPoint));

				for (i = 0; i < numprocs - 1; i++)
				{
					MPI_Recv(tempCluster, k, ClusterPointMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					MPI_Recv(points + i*partSize, partSize, PointMPIType, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
					mergeClustersWithOMP(clusters, tempCluster, k);
				}

				/*now the clusters marge but the bace point didnt change*/

				//--- set bace location with OMP ---//

				setClustersLocation(clusters, k);

				flag = 1;
				
				//we will chack the clusters stability, 
				//flag 0 = the clusters are not stability.
				//flag 1 = the clusters are stability, and we go to the next iteration.
				fflush(stdout);
				for (i = 0; i < numprocs - 1; i++)
				{
					MPI_Send(points + i*partSize, partSize, PointMPIType, i + 1, 0, MPI_COMM_WORLD);
					MPI_Send(clusters, k, ClusterPointMPIType, i + 1, 0, MPI_COMM_WORLD);

				}

				fflush(stdout);
				if (CheckClustersStability(clusters, points + (numprocs - 1)*partSize, partSize + (numberOfPoints % numprocs), k) == 0)
					flag = 0;

				for (i = 0; i < numprocs - 1; i++)
				{
					int temp;
					MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					if (temp == 0)
						flag = 0;
				}

				if (flag == 0)
				{
					for (int i = 1; i < numprocs; i++)
					{
						MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
					}
				}
				fflush(stdout);
			} while (flag == 0);

			// send tag 1 to the other procs to finish ther job in the stability.
			for (int i = 1; i < numprocs; i++)
				MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

			/*calculate the error*/
			{
				float **externalRadius = getExternalRadius(clusters, k);
				int numOfThreads = omp_get_max_threads(), numOfPointInCluster, tempClusterId, j;
				float *radius = (float*)calloc(k, sizeof(float));
				Point **tempClustersPoints = (Point**)calloc(k, sizeof(Point*)); //NEED TO FREE!!!

				for (i = 0; i < k; i++)
				{
					tempClustersPoints[i] = (Point*)calloc(clusters[i].numbersOfPoints + 1, sizeof(Point));
				}
		
				fflush(stdout);
				for (i = 0; i < numberOfPoints; i++)
				{
					int clusterIndex = points[i].clusterId;
					int currentSize = ++tempClustersPoints[clusterIndex][0].clusterId;
					tempClustersPoints[clusterIndex][currentSize] = points[i];

				}
				/*tempClustersPoints contains all the Points sort by cluster*/

				/*calculate the diameter for the clusters by omp&mpi*/
				for (i = 0; i < numprocs - 1 && i < k; i++)
				{
					numOfPointInCluster = tempClustersPoints[i][0].clusterId;
					MPI_Send(&numOfPointInCluster, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
					MPI_Send(&(tempClustersPoints[i][1]), numOfPointInCluster, PointMPIType, i + 1, 0, MPI_COMM_WORLD);
				}
				while (i < k)
				{
					MPI_Recv(&tempClusterId, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					MPI_Recv(&(radius[tempClusterId]), 1, MPI_FLOAT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
					numOfPointInCluster = tempClustersPoints[i][0].clusterId;
					MPI_Send(&numOfPointInCluster, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					MPI_Send(&(tempClustersPoints[i][1]), numOfPointInCluster, PointMPIType, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					i++;
				}

				for (i = 0; i < numprocs - 1 && i < k; i++)//for (i = 0; i < k - 1; i++)
				{

					MPI_Recv(&tempClusterId, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					MPI_Recv(&(radius[tempClusterId]), 1, MPI_FLOAT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
					numOfPointInCluster = -1;
					MPI_Send(&numOfPointInCluster, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

				}
				printf("i=%d", i);
				fflush(stdout);
				for (i = 0; i < k; i++)
				{
					printf("radius[%d] = %lf\n", i, radius[i]);
					fflush(stdout);
					for (j = 0; j < k; j++) {
						printf("externalRadius[%d][%d] = %lf\n", i, j, externalRadius[i][j]);
						fflush(stdout);
					}
				}
				
				error = calculateError(externalRadius, radius, k);
				printf("Error: %lf\n", error);
				fflush(stdout);

				freeMat((void**)externalRadius, k);
				freeMat((void**)tempClustersPoints, k);
				free(radius);
			}
			printf(" =============== cluster %d  =============== \n", iter);
			fflush(stdout);
			printClusterPoint(clusters, k);
			printf("END iter %d\n", iter);
			fflush(stdout);
			for (int i = 1; i < numprocs; i++)
			{
				MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}

		t2 = MPI_Wtime();
		printf("\n**********************Final result**********************n");
		fflush(stdout);
		
		printf("Error: %lf \ntime: %lf \nitration : %d\nRuning time: %lf \n", error, time - dt, iter - 1, t2 - t1);
		fflush(stdout);
		printClusterPoint(clusters, k);
		
		printf("\n********************************************************\n");
		fflush(stdout);

		// send tag 1 to the other procs to finish ther job.
		for (int i = 1; i < numprocs; i++)
		{
			MPI_Send(&flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}
		printOutputToFile(error, clusters, k, t2 - t1 , time - dt , iter-1);
	}
	else
	{
		int j;
		MPI_Recv(&details, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		partSize = details[0];
		k = details[1];
		numberOfPoints = details[2];
		printf("File details: partSize = =%d , k=%d, numberOfPoints=%d \n", details[0], details[1], details[2]);
		fflush(stdout);
		Point* currentPoints = (Point*)calloc(partSize, sizeof(Point));
		clusters = (ClusterPoint*)calloc(k, sizeof(ClusterPoint));

		/*In this part we will match each point to her closer cluster*/
		do {
			int numOfPointInCluster;


			do {
				flag = 1; // for that moment we didn't need to recluster 
				MPI_Recv(clusters, k, ClusterPointMPIType, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(currentPoints, partSize, PointMPIType, 0, 0, MPI_COMM_WORLD, &status);

				clusterPoints(clusters, k, currentPoints, partSize, myid, firstRound);
				
				for (j = 0; j < k; j++)
				{
					printf("my rank= %d, clusters[%d].numberOfPoints=%d\n", myid, j, clusters[j].numbersOfPoints);
					fflush(stdout);
				}
				MPI_Send(clusters, k, ClusterPointMPIType, 0, 0, MPI_COMM_WORLD);
				MPI_Send(currentPoints, partSize, PointMPIType, 0, 0, MPI_COMM_WORLD);

				MPI_Recv(currentPoints, partSize, PointMPIType, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(clusters, k, ClusterPointMPIType, 0, 0, MPI_COMM_WORLD, &status);

				if (CheckClustersStability(clusters, currentPoints, partSize, k) == 0)
					flag = 0;

				MPI_Send(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);// send the clusters stability to master
				MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				//firstRound = 1;
			} while (flag == 0);

			do
			{
				Point* pointsByCluster;
				float maxRadius;
				float* radius;
				float maxRad;
				MPI_Recv(&numOfPointInCluster, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				
				fflush(stdout);
				if (numOfPointInCluster != -1)
				{
					pointsByCluster = (Point*)calloc(numOfPointInCluster, sizeof(Point));
					radius = (float*)calloc(numOfPointInCluster, sizeof(float));

					MPI_Recv(pointsByCluster, numOfPointInCluster, PointMPIType, 0, 0, MPI_COMM_WORLD, &status);
					maxRad = calculateRadiusWithOMP(pointsByCluster, numOfPointInCluster);
				
					MPI_Send(&pointsByCluster[0].clusterId, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					MPI_Send(&maxRad, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					free(pointsByCluster);
					free(radius);
				}
			} while (numOfPointInCluster != -1);

			//the procs can know from the tag parameter if he need another iteration.  
			MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		} while (status.MPI_TAG == 0); // to finish the slaves
		free(currentPoints);
		free(clusters);
	}

	MPI_Finalize();
	return 0;
}

void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm)
{

	int offset, i;
	FILE* f = fopen("input.txt", "r");

	offset = fscanf(f, "%d", n);// N - number of points
	offset = fscanf(f, "%d", k);// K - number of clusters to find
	offset = fscanf(f, "%lf", t);// T - defines the end of time interval
	offset = fscanf(f, "%lf", dt);// dT - defines moments t = n*dT, n = { 0, 1, 2 _, T/dT} for which calculate the clusters and the quality
	offset = fscanf(f, "%d", limit);// LIMIT - the maximum number of iterations for K-MEANS algorithm
	offset = fscanf(f, "%lf", qm);// QM - quality measure to stop

	printf("n=%d, k=%d, t=%lf, dt=%lf, limit=%d, qm=%lf\n", *n, *k, *t, *dt, *limit, *qm);
	fflush(stdout);
	*points = (Point*)calloc(*n, sizeof(Point));

	for (i = 0; i < *n; i++)
	{
		offset = fscanf(f, "%f", &((*points)[i].x));
		offset = fscanf(f, "%f", &((*points)[i].y));
		offset = fscanf(f, "%f", &((*points)[i].speedX));
		offset = fscanf(f, "%f", &((*points)[i].speedY));
	}
	fclose(f);
}

ClusterPoint* creatClusters(int k, Point* p)
{
	int i;
	ClusterPoint* cluster = (ClusterPoint*)calloc(k, sizeof(ClusterPoint));

	printf("start my id :  \n");
	fflush(stdout);

#pragma omp parallel  for
	for (i = 0; i < k; i++)
	{
		cluster[i].x = p[i].x;
		cluster[i].y = p[i].y;
		cluster[i].sumX = 0;
		cluster[i].sumY = 0;
		cluster[i].numbersOfPoints = 0;
	}

	return cluster;
}

void clusterPoints(ClusterPoint* clusters, int k, Point *points,
	int numberOfPoints, int myid, int firstRound)
{
	int numOfThreads, i, j, myrank;
	ClusterPoint* temp;
	numOfThreads = omp_get_max_threads();
	temp = (ClusterPoint*)calloc(k*numOfThreads, sizeof(ClusterPoint));


	//#pragma omp parallel for
	for (i = 0; i < numberOfPoints; i++)
	{
		int index = getCluster(clusters, k, &(points[i]));
		//myrank = omp_get_thread_num();
		addPoint(&(points[i]), clusters, index, k);
	}

	free(temp);
}

void addPoint(Point* currentPoint, ClusterPoint* clusters, int currentCluster, int k)
{
	// change the new cluster values
	clusters[currentCluster].numbersOfPoints++;
	clusters[currentCluster].sumX += currentPoint->x;
	clusters[currentCluster].sumY += currentPoint->y;
	currentPoint->clusterId = currentCluster;
}

int getCluster(ClusterPoint* clusters, int k, Point* p)
{
	int i, currentCluster = 0;
	double minDistance = sqrt(pow((p->x - clusters[0].x), 2) + pow((p->y - clusters[0].y), 2));

	for (i = 1; i < k; i++)
	{
		double temp = sqrt(pow((p->x - clusters[i].x), 2) + pow((p->y - clusters[i].y), 2));

		if (minDistance > temp)
		{
			minDistance = temp;
			currentCluster = i;
		}
	}
	return currentCluster;
}
void mergeClustersWithOMP(ClusterPoint* destCluster, ClusterPoint* sourceCluster, int k)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < k; i++)
	{
		destCluster[i].numbersOfPoints += sourceCluster[i].numbersOfPoints;
		destCluster[i].sumX += sourceCluster[i].sumX;
		destCluster[i].sumY += sourceCluster[i].sumY;
	}
}

void setClustersLocation(ClusterPoint* clusters, const int k)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < k; i++)
	{
		clusters[i].x = clusters[i].sumX / clusters[i].numbersOfPoints;
		clusters[i].y = clusters[i].sumY / clusters[i].numbersOfPoints;
		printf("CLUSTER %dIS: (%f,%f)\tnumofPoints=%d\n", i, clusters[i].x, clusters[i].y, clusters[i].numbersOfPoints);
		fflush(stdout);
	}

}

/*can be parallel*/
void getMaxRadius(float **radius, Point *points, int size, int k)
{
	int i;
	float *temp = (float*)calloc(k, sizeof(float));
	for (i = 0; i < size; i++)
	{
		int clusterIndex = points[i].clusterId;
		if ((*radius)[i] > temp[clusterIndex])
			temp[clusterIndex] = (*radius)[i];
	}
	free(*radius);
	fflush(stdout);
	*radius = temp;
}
float ** getExternalRadius(ClusterPoint* clusters, int k)
{
	int i, j;
	float** radius = (float**)calloc(k, sizeof(float*));
	for (i = 0; i < k; i++)
		radius[i] = (float*)calloc(k, sizeof(float));
	for (i = 0; i < k; i++)
	{
		for (j = i + 1; j < k; j++)
		{
			radius[i][j] = (float)(pow((clusters[i].x - clusters[j].x), 2) + pow((clusters[i].y - clusters[j].y), 2));
			radius[j][i] = (float)radius[i][j];
		}
	}
	return radius;
}

/*return 0 if the cluster need to recluster and 1 else*/
int CheckClustersStability(ClusterPoint* clusters, Point *points, int size, int k)
{
	int i, currentCluster, newCluster;
	for (i = 0; i < size; i++)
	{
		currentCluster = points[i].clusterId;
		newCluster = getCluster(clusters, k, &(points[i]));
		if (currentCluster != newCluster)
		{
			printf("Point ( %f, %f) moved from ( %d ) to (%d)", points[i].x, points[i].y, currentCluster, newCluster);
			fflush(stdout);
			return 0;
		}
	}
	return 1;
}
void freeExtrnalRadius(float ** extrnalRadius, int size)
{
	int i;
	for (i = 0; i < size; i++)
		free(extrnalRadius[i]);
	free(extrnalRadius);
}

/*init the count,sumX,sumY*/
void resetClusters(ClusterPoint *c, int k)
{
	int i;
	for (i = 0; i < k; i++)
	{
		c[i].numbersOfPoints = 0;
		c[i].sumX = 0;
		c[i].sumY = 0;
	}
}
double calculateError(float **externalRadius, float *radius, int k)
{
	//q = (d1 / D12 + d1 / D13 + d2 / D21 + d2 / D23 + d3 / D31 + d3 / D32) / 6,
	int i, j;
	double sum = 0;
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < k; j++)
		{
			if (i != j)
				sum += (radius[i] / externalRadius[i][j]);
		}
	}
	for (i = 0; i < k; i++)
	{
		printf("rad[%d] = %f\n", i, radius[i]);
		fflush(stdout);
	}
	printf("==>sum = %lf\n", sum);
	fflush(stdout);

	return sum / (k*(k - 1));
}
float calculateRadiusWithOMP(Point* pointsByCluster, int size)
{
	int i, numOfThreads = omp_get_max_threads();
	float* tempRadius = (float*)calloc(numOfThreads, sizeof(float));
	float radius, maxradius = 0;
#pragma omp parallel for
	for (i = 0; i < size; i++)
	{
		int j, myRank = omp_get_thread_num();
		for (j = i + 1; j < size; j++)
		{
			radius = (float)(pow((pointsByCluster[i].x - pointsByCluster[j].x), 2) + pow((pointsByCluster[i].y - pointsByCluster[j].y), 2));
			if (radius > tempRadius[myRank])
				tempRadius[myRank] = radius;
		}
	}
	for (i = 0; i < numOfThreads; i++)
	{
		if (maxradius < tempRadius[i])
			maxradius = tempRadius[i];
	}

	free(tempRadius);
	printf(" \MyRadius = %f \n", maxradius);
	return maxradius;

}
void freeMat(void** mat, int k)
{
	int i;
	for (i = 0; i < k; i++)
		free(mat[i]);
	free(mat);
}

void printClusterPoint(ClusterPoint *clusters, int k)
{
	int i;
	for (i = 0; i < k; i++)
	{
		printf("Cluster number: %d -------> basePoint [ %f , %f ]\n", i, clusters[i].x, clusters[i].y);
		fflush(stdout);
	}
}

void printOutputToFile(double error, ClusterPoint* clusters, int k, double op_time , double time , int iter)
{
	FILE* output;
	output = fopen("output.txt", "w+");
	fprintf(output, " error = %lf operation time = %lf\n", error , op_time);
	fprintf(output, "Centers of the clusters :\n");
	for (int i = 0; i < k; i++)
	{
		fprintf(output, "cluster: %d:\t (%lf\t%lf)\n", i, clusters[i].x, clusters[i].y);
	}
	fclose(output);
}

void print_points(Point* points, int n)
{
	FILE* output;
	output = fopen("output_points.txt", "w+");
	for (int i = 0; i < n; i++)
	{
		fprintf(output, "%lf %lf %lf %lf\n", points[i].x, points[i].y, points[i].speedX, points[i].speedY);
	}
	fclose(output);
}

void print_clusters(ClusterPoint* clusters, int k)
{
	FILE* output;
	output = fopen("output_clusters.txt", "w+");
	for (int i = 0; i < k; i++)
	{
		fprintf(output, "id:%d x:%lf y:%lf numberpoints:%d, sumx:%lf sumy:%lf\n", i,clusters[i].x, clusters[i].y, clusters[i].numbersOfPoints, clusters[i].sumX, clusters[i].sumY);
	}
	fclose(output);
}
double distance(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}


