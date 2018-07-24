#pragma once
#include "Header.h"
#include<mpi.h>
#include<stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cstdlib>


#define STRUCT_SIZE_POINT 5
#define STRUCT_SIZE_CLUSTER_POINT 5

void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm);
int getCluster(ClusterPoint* clusters, int k, Point* p);
ClusterPoint* creatClusters(int k, Point* p);
void mergeClustersWithOMP(ClusterPoint* destCluster, ClusterPoint* sourceCluster, int k);
void addPoint(Point* currentPoint, ClusterPoint* clusters, int currentCluster, int firstRound);
void clusterPoints(ClusterPoint* clusters, int k, Point *points, int numberOfPoints, int myid, int firstRound);
void setClustersLocation(ClusterPoint* clusters, const int k);
/*return 0 if the cluster need to recluster and 1 else*/
int CheckClustersStability(ClusterPoint *clusters, Point *points, int clusterIndex, int k);
float ** getExternalRadius(ClusterPoint* clusters, int k);
void freeExtrnalRadius(float ** extrnalRadius, int size);
void resetClusters(ClusterPoint *c, int k);
double calculateError(float **externalRadius, float *radius, int k);
float calculateRadiusWithOMP(Point* pointsByCluster, int size);
void freeMat(void** mat, int k);
void printClusterPoint(ClusterPoint *clusters, int k);
void addPoint(Point* currentPoint, ClusterPoint* clusters, int currentCluster, int k);
void printOutputToFile(double q, ClusterPoint* clusters, int k, double op_time , double time , int iterB);
void print_points(Point* points, int n);
void print_clusters(ClusterPoint* clusters, int k);
