// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void add(float * inoutput, float * vals, int len)
{
	__shared__ float val;
	if(threadIdx.x==0)
		val = vals[(int)(blockIdx.x/2.0)-1];
	__syncthreads();

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if( blockIdx.x>1 && idx < len)
	{		
		inoutput[idx] += val;
	}

}

__global__ void scan(float * input, float * output, float * partSums, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

	__shared__ float ds_sum [2*BLOCK_SIZE];
	int blockOs = blockIdx.x*2*blockDim.x;

	if(blockOs + threadIdx.x < len)
		ds_sum[threadIdx.x] = input[blockOs + threadIdx.x];
	else
		ds_sum[threadIdx.x] = 0;

	if(blockOs + threadIdx.x + BLOCK_SIZE < len)
		ds_sum[threadIdx.x+BLOCK_SIZE] = input[blockOs + threadIdx.x + BLOCK_SIZE];
	else
		ds_sum[threadIdx.x+BLOCK_SIZE] = 0;

	for(int stride = 1; stride <= BLOCK_SIZE; stride*=2)
	{
		__syncthreads();
		int idx = (threadIdx.x+1)*stride*2 - 1;
		if(idx < 2*BLOCK_SIZE)
		{
			ds_sum[idx] += ds_sum[idx - stride];
		}
	}
	
	for(int stride = BLOCK_SIZE/2; stride >= 1; stride/=2)
	{
		__syncthreads();
		int idx = (threadIdx.x+1)*stride*2 - 1;
		if(idx + stride < 2*BLOCK_SIZE)
		{
			ds_sum[idx + stride] += ds_sum[idx];
		} 
	}

	__syncthreads();
	if(blockOs + threadIdx.x < len)
		output[blockOs + threadIdx.x] = ds_sum[threadIdx.x];
	if(blockOs + threadIdx.x + BLOCK_SIZE < len)
		output[blockOs + threadIdx.x + BLOCK_SIZE] = ds_sum[threadIdx.x+BLOCK_SIZE];
	
	if(partSums != NULL && threadIdx.x == BLOCK_SIZE-1)
		partSums[blockIdx.x] = ds_sum[threadIdx.x+BLOCK_SIZE];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * partSums;
    float * partSumsScanned;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

	dim3 dimGrid(ceil((numElements*0.5) / BLOCK_SIZE), 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);


    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	
	wbCheck(cudaMalloc((void**)&partSums, dimGrid.x*sizeof(float)));
	wbCheck(cudaMalloc((void**)&partSumsScanned, dimGrid.x*sizeof(float)));
	
	scan<<<dimGrid,dimBlock>>>( deviceInput , deviceOutput , partSums, numElements);
	
	dim3 dimGrid2(ceil((dimGrid.x*0.5)/BLOCK_SIZE), 1, 1);
	dim3 dimBlock2(BLOCK_SIZE, 1, 1);
	
    wbLog(TRACE, "DimGrid for second scan ", dimGrid2.x);
	
	scan<<<dimGrid2,dimBlock2>>>( partSums , partSumsScanned , NULL, dimGrid.x);
	
	
	dim3 dimGrid3(ceil(numElements/(float)(BLOCK_SIZE)), 1, 1);
	dim3 dimBlock3(BLOCK_SIZE, 1, 1);
	add<<<dimGrid3,dimBlock3>>>( deviceOutput , partSumsScanned , numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(partSums);
	cudaFree(partSumsScanned);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

