
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>


struct cuComplex
{
	float r;
	float i;

	__host__ __device__ cuComplex(float a, float b) : r(a), i(b) { }
	__host__ __device__ float magnitude2() { return r*r + i*i; }
	__host__ __device__ cuComplex operator*(const cuComplex & a) { return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i) ; }
	__host__ __device__ cuComplex operator+(const cuComplex & a) { return cuComplex(r+a.r,i+a.i); } 
};

__host__ __device__ bool julia(int x, int y, int dimX, int dimY)
{
	const float scale = 1.5;
	float jx = scale * (float)(dimX/2.0 - x) / (dimX/2.0);
	float jy = scale * (float)(dimY/2.0 - y) / (dimY/2.0);

	cuComplex c(-0.8,0.156);
	cuComplex a(jx,jy);

	int i = 0;
	for(i=0; i< 200; ++i)
	{
		a = a*a + c;
		if(a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

void cpuJuliaKernel(unsigned char* data, long rows, long cols)
{
	for(int r = 0; r < rows; ++r)
	{
		for(int c = 0; c < cols; ++c)
		{
			data[r*cols+c] = 255* julia(c,r, cols, rows);
		}
	}
}

__global__ void gpuJuliaKernel(unsigned char* data, long rows, long cols)
{
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.x*blockDim.x + threadIdx.x;

	if(r < rows && c < cols)
		data[r*cols+c] = 255 * julia(c,r, cols, rows);
}

int main()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int imgWidth = 1000;
	int imgHeight = 1000;
	cv::Mat1b img(imgHeight,imgWidth);
	cv::Mat1b imgGPU(imgHeight,imgWidth);
	int blockSize = 16;
	
	cpuJuliaKernel(img.data, img.rows, img.cols);

	unsigned char* dImg;
	dim3 grid( ceil(((float)imgWidth)/blockSize), ceil( ((float)imgHeight)/blockSize),1);
	dim3 threads(blockSize, blockSize, 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaMalloc((void**) &dImg, imgGPU.rows * imgGPU.cols * sizeof(unsigned char));
	gpuJuliaKernel<<<grid,threads>>>( dImg, imgGPU.rows, imgGPU.cols );
	//cudaDeviceSynchronize();
	cudaMemcpy(imgGPU.data, dImg, imgGPU.rows*imgGPU.cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(dImg);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to do Julia: %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cv::namedWindow("Julia Set");
	cv::imshow("Julia Set", imgGPU);
	cv::waitKey();
	return 0;
}

