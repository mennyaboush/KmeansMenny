#pragma once
#include "Header.h"
#define DEFULT_RADIUS 1000000
int getNumberOfBlocks(int size);
void error(Point * p);
void error(Point * p, float *r);

/*the global function insert to radius array the min value she found */
__global__ void internalRadius(int size, Point *points, float * radius)
{
	int i;
	int b = blockIdx.x;
	int t = threadIdx.x;
	int index = b*BLOCK_SIZE + t;
	for (i = index + 1; i < size; i++)
	{
		if (points[index].clusterId == points[i].clusterId)
		{
			float temp = sqrt(pow(points[index].x - points[i].x, 2) + pow(points[index].y - points[i].y, 2));
			if (temp > radius[index])
				radius[index] = temp;
		}
	}
}

/*the global function insert to radius array  0 */
__global__ void initRadius(int size, float * radius)
{
	int index = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if (index < size)
		radius[index] = 0;
}
cudaError_t internalRadiusWithCuda(int size, Point *points, float *radius)
{
	int numberOfBlocks = 0;
	Point *dev_points = 0;
	float *dev_radius = 0;
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "internalRadiusWithCuda - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fflush(stdout);
		error(points, radius);
	}

	//  cuda malloc   .
	cudaStatus = cudaMalloc((void**)&dev_points, size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "internalRadiusWithCuda -  dev_points cudaMalloc failed!");
		fflush(stdout);
		error(points, radius);
	}
	cudaStatus = cudaMalloc((void**)&dev_radius, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "internalRadiusWithCuda -  dev_radius cudaMalloc failed!");
		fflush(stdout);
		error(points, radius);
	}

	
	numberOfBlocks = getNumberOfBlocks(size);
	
	// cudaMemcpyHostToDevice.
	cudaStatus = cudaMemcpy(dev_points, points, size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "internalRadiusWithCuda - dev_points cudaMemcpy failed!(cudaMemcpyHostToDevice)");
		error(points, radius);
	}

	// insert values to dev_radius.
	internalRadius <<<numberOfBlocks, BLOCK_SIZE >>> (size, dev_points, dev_radius);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf( "internalRadiusWithCuda launch failed (internalRadius): %s\n");
		error(points, radius);
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching movePoints!\n", cudaStatus);
		error(points);

	}
	cudaStatus = cudaMemcpy(radius, dev_radius, size * sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "internalRadiusWithCuda - cudaMemcpy failed!(cudaMemcpyDeviceToHost)");
		error(points, radius);
	}
	error(points, radius);

	return cudaStatus;
}

/*the global function move the points  */
__global__ void movePoints(int size, Point *points, double Dtime)
{
	int b = blockIdx.x;
	int t = threadIdx.x;
	int index = b*BLOCK_SIZE + t;
	if (index < size)
	{
		points[index].x += Dtime*points[index].speedX;
		points[index].y += Dtime*points[index].speedY;
	}
}
cudaError_t movePointsWithCuda(int numberOfPoints, Point *points, double Dtime)
{
	int numberOfBlocks = 0;
	Point *dev_points = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		error(points);
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, numberOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - cudaMalloc failed!\n");
		error(points);

	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, numberOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - dev_points cudaMemcpy failed! (cudaMemcpyHostToDevice)\n");
		error(points);

	}

	numberOfBlocks = getNumberOfBlocks(numberOfPoints);
	movePoints  <<<numberOfBlocks, BLOCK_SIZE >>> (numberOfPoints, dev_points, Dtime);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePoints launch failed (movePoints): %s\n", cudaGetErrorString(cudaStatus));
		error(points);

	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching movePoints!\n", cudaStatus);
		error(points);

	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, numberOfPoints * sizeof(Point),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - cudaMemcpy failed!(cudaMemcpyDeviceToHost)");
		error(points);

	}
	//cudaStatus = cudaThreadExit();
	//if (cudaStatus != cudaSuccess)
	//{
	//	fprintf(stderr, "cudaThreadExit failed!");
	//	error(points);
	//}
	error(points);
	return cudaStatus;
}
int getNumberOfBlocks(int size)
{
	int num = size / BLOCK_SIZE;
	if (size % BLOCK_SIZE > 0 || num == 0) num++;
	return num;
}
void error(Point * p)
{
	cudaFree(p);
}
void error(Point * p, float *r)
{
	cudaFree(p);
	cudaFree(r);
}
