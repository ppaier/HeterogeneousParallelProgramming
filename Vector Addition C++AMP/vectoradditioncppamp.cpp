#include <wb.h> 
#include <amp.h>
using namespace concurrency;

void vectorAdd(float* hi1, float* hi2, float* ho, int n)
{
	array_view<float,1> ai1(n,hi1);
	array_view<float,1> ai2(n,hi2);
	array_view<float,1> ao(n,ho);
	ao.discard_data();

	parallel_for_each(ao.get_extent(), [=](index<1> idx) restrict(amp) 
	{ 
		ao[idx] = ai1[idx] + ai2[idx]; 
	});
	ao.synchronize();
}

int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;

	args = wbArg_read(argc, argv);

	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	
	vectorAdd(hostInput1, hostInput2, hostOutput, inputLength);

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}
