
#include "Header.h"
int getNumberOfBlocks(int size);
void error(Point * p);

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void movePoints(int size, Point *points , int Dtime)
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

cudaError_t movePointsWithCuda(int size, Point *points , int Dtime)
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
	cudaStatus = cudaMalloc((void**)&dev_points, size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - cudaMalloc failed!");
		error(points);

	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - dev_points cudaMemcpy failed!");
		error(points);

	}

	numberOfBlocks = getNumberOfBlocks(size); 
	movePoints <<<numberOfBlocks, BLOCK_SIZE>>>(size, points, Dtime);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePoints launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(points, dev_points, size * sizeof(Point), 
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "movePointsWithCuda - cudaMemcpy failed!");
		error(points);

	}
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
