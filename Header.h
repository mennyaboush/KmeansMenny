#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#define BLOCK_SIZE 1000

typedef struct
{
	float x;
	float y;
	float speedX;
	float speedY;

}Point;
Point point;

typedef struct
{
	float x;
	float y;
	int physSize;
	int logicSize;
	Point* points;
}ClusterPoint;
ClusterPoint clusterPoint;
cudaError_t movePointsWithCuda(int size, Point *points, int Dtime);
