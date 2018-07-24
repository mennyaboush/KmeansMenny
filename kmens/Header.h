#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#define BLOCK_SIZE 1000
//#define _CRT_SECURE_NO_WARNINGS

typedef struct
{
	float x;
	float y;
	float speedX;
	float speedY;
	int clusterId;
}Point;

typedef struct
{
	float x;
	float y;
	int numbersOfPoints;
	float sumX;
	float sumY;
}ClusterPoint;

static Point point;
static ClusterPoint clusterPoint;
cudaError_t movePointsWithCuda(int size, Point *points, double Dtime);
cudaError_t internalRadiusWithCuda(int size, Point *points, float *radius);