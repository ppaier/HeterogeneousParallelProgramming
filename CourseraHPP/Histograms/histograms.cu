// Histogram Equalization

#include  <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_WIDTH 16


//@@ insert code here
__global__ void imgToGray(float *imgIn, unsigned char *imgOut, 
						int rows, int cols, int channels) 
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(row < rows && col < cols)
	{
		imgOut[row*cols+col] = (unsigned char) (0.21*(255*imgIn[channels*(row*cols+col)]) +
			0.71*(255*imgIn[channels*(row*cols+col)+1]) + 0.07*(255*imgIn[channels*(row*cols+col)+2]));
	}
}

__global__ void imgHist(unsigned char *imgGray, float *hist, int rows, int cols)
{
	__shared__ float tmpHist[HISTOGRAM_LENGTH];
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int hIdx = threadIdx.x+threadIdx.y*blockDim.x;

	if( hIdx < HISTOGRAM_LENGTH)
		tmpHist[hIdx] = 0;

	__syncthreads();

	if(row < rows && col < cols)
	{
		atomicAdd(&tmpHist[imgGray[row*cols+col]],1.0/(rows*cols));
	}

	__syncthreads();
	
	if( hIdx < HISTOGRAM_LENGTH)
		atomicAdd(&hist[hIdx], tmpHist[hIdx]);
}

__global__ void scan(float * input, float * output, unsigned int * partSums, int len) {

	__shared__ float ds_sum [2*HISTOGRAM_LENGTH];
	int blockOs = blockIdx.x*2*blockDim.x;

	if(blockOs + threadIdx.x < len)
		ds_sum[threadIdx.x] = input[blockOs + threadIdx.x];
	else
		ds_sum[threadIdx.x] = 0;

	if(blockOs + threadIdx.x + HISTOGRAM_LENGTH < len)
		ds_sum[threadIdx.x+HISTOGRAM_LENGTH] = input[blockOs + threadIdx.x + HISTOGRAM_LENGTH];
	else
		ds_sum[threadIdx.x+HISTOGRAM_LENGTH] = 0;

	for(int stride = 1; stride <= HISTOGRAM_LENGTH; stride*=2)
	{
		__syncthreads();
		int idx = (threadIdx.x+1)*stride*2 - 1;
		if(idx < 2*HISTOGRAM_LENGTH)
		{
			ds_sum[idx] += ds_sum[idx - stride];
		}
	}
	
	for(int stride = HISTOGRAM_LENGTH/2; stride >= 1; stride/=2)
	{
		__syncthreads();
		int idx = (threadIdx.x+1)*stride*2 - 1;
		if(idx + stride < 2*HISTOGRAM_LENGTH)
		{
			ds_sum[idx + stride] += ds_sum[idx];
		} 
	}

	__syncthreads();
	if(blockOs + threadIdx.x < len)
		output[blockOs + threadIdx.x] = ds_sum[threadIdx.x];
	if(blockOs + threadIdx.x + HISTOGRAM_LENGTH < len)
		output[blockOs + threadIdx.x + HISTOGRAM_LENGTH] = ds_sum[threadIdx.x+HISTOGRAM_LENGTH];
	
	if(partSums != NULL && threadIdx.x == HISTOGRAM_LENGTH-1)
		partSums[blockIdx.x] = ds_sum[threadIdx.x+HISTOGRAM_LENGTH];
}

__global__ void equHistogram(float *imgIn, float *imgOut, float *cdf, int rows, int cols, int channels)
{
	__shared__ float cdfMin;
	if(threadIdx.x == 0 && threadIdx.y==0)
		cdfMin = cdf[0];

	__syncthreads();

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(row < rows && col < cols)
	{
		unsigned char val = (unsigned char) 255*imgIn[channels*(row*cols+col)];		
		imgOut[channels*(row*cols+col)] = (cdf[val] - cdfMin)/(1 - cdfMin);

		val = (unsigned char) 255*imgIn[channels*(row*cols+col)+1];		
		imgOut[channels*(row*cols+col)+1] = (cdf[val] - cdfMin)/(1 - cdfMin);

		val = (unsigned char) 255*imgIn[channels*(row*cols+col)+2];		
		imgOut[channels*(row*cols+col)+2] = (cdf[val] - cdfMin)/(1 - cdfMin);
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
    float * deviceInputImageData;
    unsigned char * deviceInputImageDataGray;
    float * deviceOutputImageData;
	float * deviceHistogram;
	float * deviceCumHistogram;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
	
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);


	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceInputImageDataGray, imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(float));
	cudaMalloc((void **) &deviceCumHistogram, HISTOGRAM_LENGTH * sizeof(float));
	
	cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(float));

	cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);

	dim3 dimGrid ((imageWidth-1) / BLOCK_WIDTH + 1, (imageHeight-1) / BLOCK_WIDTH + 1, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH,1);
	imgToGray<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceInputImageDataGray, 
		imageHeight, imageWidth, imageChannels);

	imgHist<<<dimGrid,dimBlock>>>(deviceInputImageDataGray, deviceHistogram, 
		imageHeight, imageWidth);
		
	dim3 dimGrid2(1, 1, 1);
	dim3 dimBlock2(HISTOGRAM_LENGTH, 1, 1);
		
	scan<<<dimGrid2,dimBlock2>>>( deviceHistogram , deviceCumHistogram , NULL, HISTOGRAM_LENGTH);

	equHistogram<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceCumHistogram, imageHeight, imageWidth, imageChannels);
	
	cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * sizeof(float) * imageChannels,
               cudaMemcpyDeviceToHost);

    wbSolution(args, outputImage);

    //@@ insert code here

    cudaFree(deviceInputImageData);
    cudaFree(deviceInputImageDataGray);
    cudaFree(deviceOutputImageData);
	cudaFree(deviceHistogram);
	cudaFree(deviceCumHistogram);
	
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

