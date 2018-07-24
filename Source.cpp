#include "Header.h"
#include<mpi.h>
#include<stdlib.h>

#define STRUCT_SIZE_POINT 4
#define STRUCT_SIZE_CLUSTER_POINT 5
#define DSIZE 100
int main(int argc, char * argv[]);

void printPoints(Point * points, int size);
void readFromFile(Point** points, int* n, int* k, int* t, double* dt,
					int* limit, double* qm);
ClusterPoint* creatClusters(int k);
int main(int argc, char *argv[])
{
	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int numberOfPoints, k, maxTime, limitIter;
	double qm, dt ,newX , newY;
	Point *points = NULL;
	ClusterPoint * clusters;
	int partSize = numberOfPoints / numprocs;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Get_processor_name(processor_name, &namelen);

	MPI_Status status;
#pragma region createDataType
	MPI_Datatype PointMPIType;
	MPI_Datatype type[STRUCT_SIZE_POINT] 
		= { MPI_FLOAT,MPI_FLOAT,MPI_FLOAT ,MPI_FLOAT};
	int blocklen[STRUCT_SIZE_POINT] = {1, 1, 1 ,1};
	MPI_Aint disp[STRUCT_SIZE_POINT];
	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.speedX - (char *)&point;
	disp[3] = (char *)&point.speedY - (char *)&point;
	MPI_Type_create_struct(STRUCT_SIZE_POINT, blocklen, disp,
		type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);
	


	MPI_Datatype ClusterPointMPIType;
	MPI_Datatype type[STRUCT_SIZE_CLUSTER_POINT]
		= { MPI_FLOAT,MPI_FLOAT,MPI_INT ,MPI_INT , PointMPIType
	};
	int blocklen[STRUCT_SIZE_CLUSTER_POINT] = { 1, 1, 1 ,1, 1};
	MPI_Aint disp[STRUCT_SIZE_POINT];
	disp[0] = (char *)&clusterPoint.x - (char *)&clusterPoint;
	disp[1] = (char *)&clusterPoint.y - (char *)&clusterPoint;
	disp[2] = (char *)&clusterPoint.logicSize - (char *)&clusterPoint;
	disp[3] = (char *)&clusterPoint.physSize - (char *)&clusterPoint;
	disp[4] = (char *)&clusterPoint.points - (char *)&clusterPoint;
	MPI_Type_create_struct(STRUCT_SIZE_POINT, blocklen, disp,
		type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);

#pragma endregion
	clusters = creatClusters(k);

	if (myid == 0)
	{
		int i;
		newX = newY = 0;
		readFromFile(&points, &numberOfPoints, &k, &maxTime, &dt, &limitIter, &qm);
		for (i = 1; i < numprocs; i++)
		{
			MPI_Send(points + i*partSize, partSize, PointMPIType, i, 0, MPI_COMM_WORLD);
		}
		clusterPointsWhithMpi(clusters, k, points+(numprocs-1)*partSize, partSize+ (numberOfPoints % numprocs));
		//setBaceLocation(clusters, k);
		ClusterPoint* tempCluster = (ClusterPoint*)calloc(k, sizeof(ClusterPoint));
		for (i = 1; i < numprocs; i++)
		{
			MPI_Recv(tempCluster, k, ClusterPointMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			mergeClustersWithOMP(clusters, tempCluster, k);
		}

		//--- set bace location whit MPI ---//
		for (i = 1; i < numprocs; i++)
		{
			MPI_Send(clusters[i].points, clusters[i].logicSize, PointMPIType, i, 0, MPI_COMM_WORLD);
		}
		while (i < k)
		{
			double tempBace[2];
			MPI_Recv(tempBace, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			clusters[i].x = tempBace[0] / clusters[i].logicSize;
			clusters[i].y = tempBace[1] / clusters[i].logicSize;
			MPI_Send(clusters[i].points, clusters[i].logicSize, PointMPIType, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			i++;
		}
		for (i = 1; i < numprocs; i++)
		{
			double tempBace[2];
			MPI_Recv(tempBace, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			newX += tempBace[0];
			newY += tempBace[1];
			MPI_Send(clusters[i].points, clusters[i].logicSize, PointMPIType, status.MPI_SOURCE, 1, MPI_COMM_WORLD);

		}

	}
	else
	{
		Point* currentPoints=(Point*)calloc(partSize, sizeof(Point));
		MPI_Recv(currentPoints, partSize, PointMPIType, myid, 0, MPI_COMM_WORLD, &status);
		clusterPointsWhithMpi(clusters, k, currentPoints, numberOfPoints);
		//setBaceLocation(clusters, k);
		MPI_Send(clusters, k, ClusterPointMPIType, myid, 0, MPI_COMM_WORLD);
		free(currentPoints);
		while (status.MPI_TAG == 0)
		{
			MPI_Recv(temp, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			getAvg(temp)
		}
	}

		//cudaError_t cudaStatus = movePointsWithCuda(3, points, 1);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "addWithCuda failed!");
		//	return 1;
		//}

#pragma region cudaExemple

	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}
#pragma endregion

	MPI_Finalize();
	return 0;
}

void printPoints(Point * points, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("%.2f\t%.2f\t%.2f\t%.2f\n", points[i].x, points[i].y, points[i].speedX, points[i].speedY);
	}
}
void readFromFile(Point** points, int* n, int* k, int* t, double* dt, int* limit, double* qm)
{
	FILE* f = fopen("input.txt", "r");
	int offset;
	offset = fscanf(f, "%d", n);// N - number of points
	offset = fscanf(f, "%d", k);// K - number of clusters to find
	offset = fscanf(f, "%d", t);// T - defines the end of time interval
	offset = fscanf(f, "%lf", dt);// dT - defines moments t = n*dT, n = { 0, 1, 2 _, T/dT} for which calculate the clusters and the quality
	offset = fscanf(f, "%d", limit);// LIMIT - the maximum number of iterations for K-MEANS algorithm
	offset = fscanf(f, "%lf", qm);// QM - quality measure to stop

	*points = (Point*)calloc(*n ,sizeof(Point));

	for (int i = 0; i < *n; i++)
	{
		offset = fscanf(f, "%f", &((*points)[i].x));
		offset = fscanf(f, "%f", &((*points)[i].y));
		offset = fscanf(f, "%f", &((*points)[i].speedX));
		offset = fscanf(f, "%f", &((*points)[i].speedY));
	}
	fclose(f);
}
ClusterPoint* creatClusters(int k)
{
	int i;
	const int maxVal = 100;
	ClusterPoint* cluster = (ClusterPoint*)calloc(k, sizeof(ClusterPoint));
#pragma parllel for
	for (i = 0; i < k; i++)
	{
		cluster[i].points = (Point*)calloc(100, sizeof(Point));
		cluster[i].x = (float)rand() / (float)(maxVal);
		cluster[i].y = (float)rand() / (float)(maxVal);
		cluster[i].logicSize = 0;
		cluster[i].physSize = 100;
	}
	return cluster;
}
void clusterPointsWhithMpi(ClusterPoint* clusters, int k, Point *points, 
	int numberOfPoints )
{
	int i;
	for (i = 0; i < numberOfPoints; i++)
	{
		int index=getCluster(clusters, k, &points[i]);
		addPoint(&points[i], clusters, index);
	}
}
void addPoint(Point* currentPoint, ClusterPoint* clusters, int currentCluster)
{
	if (clusters[currentCluster].logicSize == clusters[currentCluster].physSize)
	{
		clusters[currentCluster].physSize += DSIZE;
		clusters[currentCluster].points = (Point*)realloc(clusters[currentCluster].points, clusters[currentCluster].physSize * sizeof(Point));
	}
	clusters[currentCluster].points[clusters[currentCluster].logicSize] = *currentPoint;
	clusters[currentCluster].logicSize++;
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
	int i,j,index, clusterSize;
	for (i = 0; i < k; i++)
	{
		clusterSize = destCluster[i].logicSize + sourceCluster[i].logicSize;
		destCluster[i].points = (Point*)realloc(destCluster[i].points, clusterSize * sizeof(Point));
#pragma parllel for
		for (j = 0,index = destCluster[i].logicSize; j < sourceCluster[i].logicSize; j++,index++)
		{
			destCluster[i].points[index] = sourceCluster[i].points[j];
		}
		destCluster[i].logicSize = destCluster[i].physSize = clusterSize;
		
	}
}
void setBaceLocation(ClusterPoint* clusters, int k)
{
	int i, j;
	float sumX, sumY;
	for (i = 0; i < k; i++)
	{
		sumX = 0;
		sumY = 0;
#pragma parllel for
		for (j = 0; j < clusters[i].logicSize; j++)
		{
			sumX += clusters[i].points[j].x;
			sumY += clusters[i].points[j].y;
		}
		if (clusters[i].logicSize > 0)
		{
			clusters[i].x = sumX / clusters[i].logicSize;
			clusters[i].y = sumY / clusters[i].logicSize;
		}
	}
}