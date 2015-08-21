#include	<wb.h>

#define BLOCK_SIZE 256
#define SEG_SIZE 1536


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//@@ Insert code to implement vector addition here	
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if(idx < len)
	{
		out[idx] = in1[idx] + in2[idx];
	}
}

int main(int argc, char ** argv) {
	wbArg_t args;
	int inputLength;

	float * hostInput1;
	float * hostInput2;
	float * hostOutput;

	float * deviceInput0a;
	float * deviceInput1a;
	float * deviceInput2a;
	float * deviceInput3a;

	float * deviceInput0b;
	float * deviceInput1b;
	float * deviceInput2b;
	float * deviceInput3b;

	float * deviceOutput0;
	float * deviceOutput1;
	float * deviceOutput2;
	float * deviceOutput3;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *) malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");
	wbTime_start(GPU, "Allocating GPU memory.");

	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	//@@ Allocate GPU memory here
	int lengthInBytes = inputLength * sizeof(float);
	int lengthChunk   = SEG_SIZE* sizeof(float);
	
	cudaHostRegister(hostInput1,lengthInBytes,0);
	cudaHostRegister(hostInput2,lengthInBytes,0);
	cudaHostRegister(hostOutput,lengthInBytes,0);

	cudaMalloc((void**) &deviceInput0a, lengthChunk);
	cudaMalloc((void**) &deviceInput0b, lengthChunk);
	cudaMalloc((void**) &deviceInput1a, lengthChunk);
	cudaMalloc((void**) &deviceInput1b, lengthChunk);	
	cudaMalloc((void**) &deviceInput2a, lengthChunk);
	cudaMalloc((void**) &deviceInput2b, lengthChunk);
	cudaMalloc((void**) &deviceInput3a, lengthChunk);
	cudaMalloc((void**) &deviceInput3b, lengthChunk);

	cudaMalloc((void**) &deviceOutput0, lengthChunk);
	cudaMalloc((void**) &deviceOutput1, lengthChunk);
	cudaMalloc((void**) &deviceOutput2, lengthChunk);
	cudaMalloc((void**) &deviceOutput3, lengthChunk);

	wbTime_stop(GPU, "Allocating GPU memory.");
	int n = ceil(((double)inputLength)/SEG_SIZE);

	wbTime_start(GPU, "CUDA Streaming.");
	for(int i=0; i<n*SEG_SIZE; i+=SEG_SIZE*4)
	{
		//@@ Copy memory to the GPU here

		int lengthChunk0 = sizeof(float) * (i+  SEG_SIZE <= inputLength ? SEG_SIZE : inputLength - i);
		int lengthChunk1 = sizeof(float) * (i+2*SEG_SIZE <= inputLength ? SEG_SIZE : inputLength - i - SEG_SIZE);
		int lengthChunk2 = sizeof(float) * (i+3*SEG_SIZE <= inputLength ? SEG_SIZE : inputLength - i - 2*SEG_SIZE);
		int lengthChunk3 = sizeof(float) * (i+4*SEG_SIZE <= inputLength ? SEG_SIZE : inputLength - i - 3*SEG_SIZE);

		if(lengthChunk0>0)
		{
			cudaMemcpyAsync(deviceInput0a, hostInput1+i, lengthChunk0, cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(deviceInput0b, hostInput2+i, lengthChunk0, cudaMemcpyHostToDevice, stream0);
		}
		
		if(lengthChunk1>0)
		{
			cudaMemcpyAsync(deviceInput1a, hostInput1+i+SEG_SIZE, lengthChunk1, cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(deviceInput1b, hostInput2+i+SEG_SIZE, lengthChunk1, cudaMemcpyHostToDevice, stream1);
		}
		
		if(lengthChunk2>0)
		{
			cudaMemcpyAsync(deviceInput2a, hostInput1+i+2*SEG_SIZE, lengthChunk2, cudaMemcpyHostToDevice, stream2);
			cudaMemcpyAsync(deviceInput2b, hostInput2+i+2*SEG_SIZE, lengthChunk2, cudaMemcpyHostToDevice, stream2);
		}

		if(lengthChunk3>0)
		{
			cudaMemcpyAsync(deviceInput3a, hostInput1+i+3*SEG_SIZE, lengthChunk3, cudaMemcpyHostToDevice, stream3);
			cudaMemcpyAsync(deviceInput3b, hostInput2+i+3*SEG_SIZE, lengthChunk3, cudaMemcpyHostToDevice, stream3);
		}

		//@@ Launch the GPU Kernel here
		if(lengthChunk0>0)
			vecAdd<<<ceil(((double)lengthChunk0)/(sizeof(float)*BLOCK_SIZE)),BLOCK_SIZE,0,stream0>>>(deviceInput0a, deviceInput0b, deviceOutput0, lengthChunk0/sizeof(float));

		if(lengthChunk1>0)
			vecAdd<<<ceil(((double)lengthChunk1)/(sizeof(float)*BLOCK_SIZE)),BLOCK_SIZE,0,stream1>>>(deviceInput1a, deviceInput1b, deviceOutput1, lengthChunk1/sizeof(float));

		if(lengthChunk2>0)
			vecAdd<<<ceil(((double)lengthChunk2)/(sizeof(float)*BLOCK_SIZE)),BLOCK_SIZE,0,stream2>>>(deviceInput2a, deviceInput2b, deviceOutput2, lengthChunk2/sizeof(float));

		if(lengthChunk3>0)
			vecAdd<<<ceil(((double)lengthChunk3)/(sizeof(float)*BLOCK_SIZE)),BLOCK_SIZE,0,stream3>>>(deviceInput3a, deviceInput3b, deviceOutput3, lengthChunk3/sizeof(float));

		//@@ Copy the GPU memory back to the CPU here
		
		if(lengthChunk0>0)
			cudaMemcpyAsync(hostOutput+i, deviceOutput0, lengthChunk0, cudaMemcpyDeviceToHost, stream0);

		if(lengthChunk1>0)
			cudaMemcpyAsync(hostOutput+i+SEG_SIZE, deviceOutput1, lengthChunk1, cudaMemcpyDeviceToHost, stream1);

		if(lengthChunk2>0)
			cudaMemcpyAsync(hostOutput+i+2*SEG_SIZE, deviceOutput2, lengthChunk2, cudaMemcpyDeviceToHost, stream2);

		if(lengthChunk3>0)
			cudaMemcpyAsync(hostOutput+i+3*SEG_SIZE, deviceOutput3, lengthChunk3, cudaMemcpyDeviceToHost, stream3);
	}
	cudaDeviceSynchronize();
	cudaHostUnregister(hostInput1);
	cudaHostUnregister(hostInput2);
	cudaHostUnregister(hostOutput);
	wbTime_stop(GPU, "CUDA Streaming.");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaFree(deviceInput0a);
	cudaFree(deviceInput1a);
	cudaFree(deviceInput2a);
	cudaFree(deviceInput3a);
	
	cudaFree(deviceInput0b);
	cudaFree(deviceInput1b);
	cudaFree(deviceInput2b);
	cudaFree(deviceInput3b);

	cudaFree(deviceOutput0);
	cudaFree(deviceOutput1);
	cudaFree(deviceOutput2);
	cudaFree(deviceOutput3);

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}

