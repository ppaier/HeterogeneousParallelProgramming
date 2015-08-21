#include <wb.h> //@@ wb include opencl.h for you

//@@ OpenCL Kernel

const char* vaddsrc = "__kernel void vadd(__global float* dA, __global float *dB, __global float* dC, int N)"
	"{\n "
	"	int id = get_global_id(0); \n "
	"	if(id<N)\n "
	"		dC[id] = dA[id] + dB[id]; \n " 
	"} " ;

int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	cl_mem deviceInput1;
	cl_mem deviceInput2;
	cl_mem deviceOutput;
		
	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = ( float * )malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	cl_int clerr = CL_SUCCESS;
	cl_uint numPlatforms;

	clerr = clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id* platforms = new cl_platform_id[numPlatforms];

	clerr = clGetPlatformIDs(numPlatforms, platforms, NULL);
	cl_context_properties properties[] =  {	CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0};

	cl_context clctx;
	clctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);

	size_t parmsz;
	clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);

	cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
	clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);

	cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);

	delete [] platforms;

	cl_program clpgm;
	clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr);

	char clcompileflags[4096];
	sprintf(clcompileflags, "-cl-mad-enable");
	clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);

	cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here

	int size = inputLength * sizeof(float);

	deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput1, &clerr);
	deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput2, &clerr);
	deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, size, NULL, &clerr);

	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	
	// already done during allocation 

	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	clerr= clSetKernelArg(clkern, 0, sizeof(cl_mem),(void *)&deviceInput1);
	clerr= clSetKernelArg(clkern, 1, sizeof(cl_mem),(void *)&deviceInput2);
	clerr= clSetKernelArg(clkern, 2, sizeof(cl_mem),(void *)&deviceOutput);
	clerr= clSetKernelArg(clkern, 3, sizeof(int), &inputLength);

	cl_event event = NULL;

	const size_t bsz = 256;
	const size_t gsz = bsz * ( (int) ( (inputLength-1)/bsz + 1 ) );

	wbTime_start(Compute, "Performing computation");
	//@@ Launch the GPU Kernel here

	clerr= clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL,
		&gsz, &bsz, 0, NULL, &event);

	std::cout << clerr << std::endl;
	clerr= clWaitForEvents(1, &event);
	wbTime_stop(Compute, "Performing computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0,
		inputLength*sizeof(float), hostOutput, 0, NULL, NULL);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here

	clFlush(clcmdq);	
	clFinish(clcmdq);
	clReleaseKernel(clkern);
	clReleaseProgram(clpgm);
	clReleaseCommandQueue(clcmdq);
	clReleaseContext(clctx);

	clReleaseMemObject(deviceInput1);
	clReleaseMemObject(deviceInput2);
	clReleaseMemObject(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}
