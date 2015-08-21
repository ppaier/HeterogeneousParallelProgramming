
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#define MAXSPHERES 20
#define INF 2e10f
#define rnd( x ) (x * rand() / RAND_MAX)

struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;

	__host__ __device__ float hit(float ox, float oy, float *n) 
	{ 
		float dx = ox - x;
		float dy = oy - y;

		if(dx*dx + dy*dy < radius * radius)
		{
			float dz = sqrtf(radius*radius - dx*dx -dy*dy);
			*n = dz / sqrtf(radius*radius);
			return dz + z;
		}
		return -INF;
	}

};

__constant__ Sphere dSpheres[MAXSPHERES];

__global__ void RayTracerKernel(unsigned char* data, long rows, long cols, long numSpheres)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	float ox = (x - cols/2.0);
	float oy = (y - rows/2.0); 

	float r = 0;
	float g = 0;
	float b = 0;

	float maxz = -INF;

	for(int i=0; i<numSpheres; ++i)
	{
		float n, t = dSpheres[i].hit(ox, oy, &n);
		if(t>maxz)
		{
			float fscale = n;
			r = dSpheres[i].r * fscale;
			g = dSpheres[i].g * fscale;
			b = dSpheres[i].b * fscale;
			maxz = t;
		}
	}

	if(y < rows && x < cols)
	{
		data[4*(y*cols+x)]   = (int) (b*255);
		data[4*(y*cols+x)+1] = (int) (g*255);
		data[4*(y*cols+x)+2] = (int) (r*255);
		data[4*(y*cols+x)+3] = 255;
	}
}

int main()
{
	int imgWidth = 800;
	int imgHeight = 640;
	int numSpheres = 20;
	numSpheres = (numSpheres <= MAXSPHERES) ? numSpheres : MAXSPHERES;
	int blockSize = 16;

	cv::Mat4b imgGPU(imgHeight,imgWidth);
	unsigned char* dImg;
		
	Sphere *tmpSpheres = (Sphere*) malloc(numSpheres*sizeof(Sphere));
	for(int i=0; i<numSpheres; ++i)
	{
		tmpSpheres[i].r = rnd(1.0f);
		tmpSpheres[i].g = rnd(1.0f);
		tmpSpheres[i].b = rnd(1.0f);
		tmpSpheres[i].x = rnd(1000.0f) - 500;
		tmpSpheres[i].y = rnd(1000.0f) - 500;
		tmpSpheres[i].z = rnd(1000.0f) - 500;
		tmpSpheres[i].radius = rnd( 100.0f ) + 20;
	}


	dim3 grid(ceil(((float)imgWidth)/blockSize),ceil(((float)imgHeight)/blockSize),1);
	dim3 threads(blockSize, blockSize, 1);

	// CUDA part start
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
		
	cudaMalloc((void**) &dImg, imgGPU.rows * imgGPU.cols * sizeof(unsigned char) * 4);
	cudaMemcpyToSymbol(dSpheres, tmpSpheres, sizeof(Sphere) * numSpheres);
	RayTracerKernel<<<grid,threads>>>( dImg, imgGPU.rows, imgGPU.cols, numSpheres );
	cudaDeviceSynchronize();
	cudaMemcpy(imgGPU.data, dImg, imgGPU.rows*imgGPU.cols*sizeof(unsigned char) * 4, cudaMemcpyDeviceToHost);
	cudaFree(dImg);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to do Raytracing: %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(tmpSpheres);

	cv::namedWindow("Spheres");
	cv::imshow("Spheres", imgGPU);
	cv::waitKey();
	return 0;
}

