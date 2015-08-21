#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define Tile_width 12
#define Block_width Tile_width + Mask_width-1
#define Max_channels 3

//@@ INSERT CODE HERE
__global__ void imgConv(float *imgIn, const float * __restrict__ mask, float *imgOut, 
						int rows, int cols, int channels, int step) 
{

	__shared__ float ds_roi [Max_channels][Block_width][Block_width];
	
	int row = blockIdx.y*Tile_width + threadIdx.y;
	int col = blockIdx.z*Tile_width + threadIdx.z;
	int row_i = row - (int)Mask_radius;
	int col_i = col - (int)Mask_radius;

	if(row_i >= 0 && col_i>= 0 && row_i < rows && col_i < cols)
		ds_roi[threadIdx.x][threadIdx.y][threadIdx.z] = imgIn[(row_i*cols+col_i)*channels + threadIdx.x];
	else
		ds_roi[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0;

	__syncthreads();

	if(threadIdx.y < Tile_width && threadIdx.z< Tile_width)
	{
		float val = 0.0;
		for(int r = 0; r<Mask_width; ++r)
		{
			for(int c=0; c<Mask_width; ++c)
			{
				val += mask[r*Mask_width + c] * ds_roi[threadIdx.x][threadIdx.y+r][threadIdx.z+c]; 
			}
		} 
		if(row < rows && col < cols)
			imgOut[(row*cols+col)*channels + threadIdx.x] = val;
	}


}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE	
	dim3 dimGrid (1,(imageHeight-1)/Tile_width+1,(imageWidth-1)/Tile_width+1);
	dim3 dimBlock(imageChannels, Block_width, Block_width);
	imgConv<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, 
		imageHeight, imageWidth, imageChannels, imageWidth*imageChannels);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
