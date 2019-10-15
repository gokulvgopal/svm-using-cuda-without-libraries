#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
//#include <chrono>
#include <cstdio>
#include <cstdlib>

//#include <thrust\device_vector.h>
//#include <thrust\sort.h>
//#include <thrust\sequence.h>
//#include <thrust\gather.h>


using namespace std;

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << "in" << file << " at line " << line;
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));
//#define maxSize 2
//#define maxQuantity 6

// data structure to store the test data, size and number of records
struct Data {
	int* value;
	int quantity;
	int size;
};

// data structure to store the possible variation of weights matrix
struct Transform {
	int* value;
	int size;
};

// data structure to store the weights and its size
struct Weights {
	float* value;
	int size;
};

// data structure to store all possible variation of ||w||, w and b that satisfy the equation.
// this data structure is also used to store the least ||w|| that will be used to predict the values.
struct Optimal {
	float* magnitude;
	float* w;
	float* b;
	int size;
	int quantity;

	Optimal(int size_, int quantity_) {
		size = size_;
		quantity = quantity_;
		cudaMalloc((void**)&magnitude, quantity * sizeof(int));
		cudaMalloc((void**)&b, quantity * sizeof(int));
		cudaMalloc((void**)&w, quantity * size * sizeof(int));
	}
};

int findMax(Data* data);

//to print the results from prediction in GPU
void d_printPrediction(float* result, int quantity) {
	cout << endl << "GPU - The given prediction set belong to following classes: { ";
	for (int i = 0; i < quantity; i++)
		cout << result[i] << ", ";
	cout << "}" << endl;
}

// function used for prediction. It is done using the formula x.w+b and result is -1 if the equation is less than 0
// otherwise result is 1.
__global__  void gpuMultiplication(float* matrixC, float* matrixA, float* matrixB, int rowA, int colA, int colB, float optimal_b) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= rowA || j >= colB)
		return;

	float sum = 0;
	for (int k = 0; k < colA; k++) {
		sum += matrixA[i*colA + k] * matrixB[k*colB + j];
	}
	sum = sum + optimal_b;
	if (sum < 0)
		matrixC[i * colB + j] = -1;
	else
		matrixC[i * colB + j] = 1;

}

// this function is to define how the predict gpu kernel needs to be called.
// It is also used to transfer all cpu variables to GPU values.
// Finally it invokes the print prediction for printing out the result.
void d_callpredict(Data* data, Optimal* optimal) {
	dim3 threads(32, 32);
	dim3 blocks(data->quantity / threads.x + 1, optimal->quantity / threads.y + 1);
	float* d_data;
	float* temp_d_data = (float*)malloc(sizeof(float) * data->quantity * data->size);
	float* result = (float*)malloc(sizeof(float) * data->quantity);
	float* d_result;
	float* d_optimal_w;
																							//cout << " data " << endl;
	for (int i = 0; i < data->quantity * data->size; i++) {
		temp_d_data[i] = data->value[i];
																							//cout << temp_d_data[i] << " x ";
	}
																							// printing weights to check if they are successfully recieved in this function
																							/*cout << endl;
																							cout << "weights: " << endl;
																							for (int i = 0; i < optimal->quantity * optimal->size; i++)
																								cout << optimal->w[i] << " x ";
																							cout << endl;*/
	HANDLE_ERROR(cudaMalloc(&d_data, data->quantity * data->size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&d_result, data->quantity * optimal->size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&d_optimal_w, optimal->quantity * optimal->size * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_data, temp_d_data, data->quantity * data->size * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_optimal_w, optimal->w, optimal->quantity * optimal->size * sizeof(float), cudaMemcpyHostToDevice));
	gpuMultiplication << <blocks, threads >> > (d_result, d_data, d_optimal_w, data->quantity, data->size, optimal->quantity, optimal->b[0]);
	HANDLE_ERROR(cudaMemcpy(result, d_result, data->quantity * optimal->quantity * sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(d_result);
	cudaFree(d_data);
	d_printPrediction(result, data->quantity);

}

// this GPU function is used to find the least index of the least amplitude 
// returns the index of that particular cell
void d_findMinOptimalIndex(float* optimal_magnitude, int index, int* optimal_min_index) {
	float* d_optimal_magnitude;
	float* temp_d_optimal_magnitude = (float*) malloc(sizeof(float)*index);
	optimal_min_index = (int*)malloc(sizeof(int));
	cublasHandle_t handle;
	cublasStatus_t stat;
	HANDLE_ERROR(cudaMalloc(&d_optimal_magnitude, sizeof(float) * index));
	HANDLE_ERROR(cudaMemcpy(d_optimal_magnitude, temp_d_optimal_magnitude, sizeof(float) * index, cudaMemcpyHostToDevice));
	cublasCreate(&handle);
	cublasIsamin(handle, index, d_optimal_magnitude, 1, optimal_min_index);
	cudaFree(d_optimal_magnitude);
	cublasDestroy(handle);

}

// this CPU function invokes the GPU cublas funcion above.
// this function is also used to copy to and from device memory
void d_findMinOptimal(Optimal* optimal, Optimal* d_optimal_best, int index) {
	int* optimal_min_index = 0;
	d_optimal_best->size = optimal->size;
	d_optimal_best->quantity = 1;
	d_optimal_best->magnitude = (float*)malloc(sizeof(float));
	d_optimal_best->b = (float*)malloc(sizeof(float));
	d_optimal_best->w = (float*)malloc(d_optimal_best->size * sizeof(float));
	float* optimal_magnitude = (float*)malloc(sizeof(float)*index);
	for (int i = 0; i < index; i++) {
		optimal_magnitude[i] = optimal->magnitude[i];
	}
	d_findMinOptimalIndex(optimal_magnitude, index, optimal_min_index);
}

// this GPU function is to get all optimal combination of w0, w1, b and magnitude.
// it then returns arrays of these values
__global__ void d_getAllOptimal(float w0_min, float w1_min, float step_w0, float step_w1, float b_min, float step_b, int size,
	int quantity, float* data, float* result, float* optimal_magnitude, float* optimal_w, float* optimal_b, int index) {
	int w0i = blockIdx.x * blockDim.x + threadIdx.x;
	int w1i = blockIdx.y * blockDim.y + threadIdx.y;
	int bi = blockIdx.z * blockDim.z + threadIdx.z;
	__shared__ int indexArr[1000*size* sizeof(int)];
	for(int m = 0; m < size*1000; m++)
		indexArr[m] = m;
	__syncthreads();
	float w0 = w0_min + w0i * step_w0;
	float w1 = w1_min + w1i * step_w1;
	float b = b_min + bi * step_b;
	bool flag = true;

	for (int i = 0; i < quantity; i++) {
		float sum = 0;
		sum = w0 * data[i*size] + w1 * data[i*size + 1] + b;
		sum *= result[i];
		if (sum >= 1)
			flag = false;
	}

	if (flag) {
		optimal_w[index*size] = w0;
		optimal_w[index*size + 1] = w1;
		optimal_b[index] = b;
		optimal_magnitude[index] = fabsf(sqrtf(pow(w0, 2) + pow(w1, 2)));
		index = index + 1;
	}

	__syncthreads();

}

// this CPU function is used to invoke the above GPU function to find all optimal cases.
// this function is also responsible for copying from and to the device memory
void d_fit(Data* data, Data* result, Optimal* optimal, Optimal* optimal_best) {
	int max = findMax(data);
	float step_sizes[] = { max*(float)0.1, max*(float)0.01, max*(float)0.001 };
	int b_multiple_range = 5;
	int b_multiple = 5;
	float latest_optimum = (float)max * (float)3;
	optimal->size = data->size;
	optimal->quantity = data->quantity;
	optimal->magnitude = (float*)malloc(1000 * optimal->quantity * sizeof(float));
	optimal->b = (float*)malloc(10000 * optimal->quantity * sizeof(float));
	optimal->w = (float*)malloc(10000 * optimal->quantity * optimal->size * sizeof(float));
	optimal_best->size = data->size;
	optimal_best->quantity = 1;
	optimal_best->magnitude = (float*)malloc(sizeof(float));
	optimal_best->b = (float*)malloc(sizeof(float));
	optimal_best->w = (float*)malloc(optimal_best->size * sizeof(float));
	optimal_best->magnitude[0] = (float)1.5 * latest_optimum;
	optimal_best->b[0] = (float)max * b_multiple_range;
	optimal_best->w[0] = latest_optimum;
	optimal_best->w[1] = latest_optimum;



	float* d_data;
	float* d_result;
	float* d_optimal_magnitude;
	float* d_optimal_w;
	float* d_optimal_b;
	int* optimal_min_index = 0;

	HANDLE_ERROR(cudaMalloc(&d_data, data->quantity * data->size * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_result, result->quantity * result->size * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_optimal_magnitude, 10000 * optimal->quantity * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&d_optimal_w, 10000 * optimal->quantity * optimal->size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&d_optimal_b, 10000 * optimal->quantity * sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(d_data, data->value, data->quantity * data->size * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_result, result->value, result->quantity * result->size * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_optimal_magnitude, optimal->magnitude, 10000 * optimal->quantity * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_optimal_w, optimal->w, 10000 * optimal->quantity * optimal->size * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_optimal_b, optimal->b, 10000 * optimal->quantity * sizeof(float), cudaMemcpyHostToDevice));

	for (int step_size_index = 0; step_size_index < 3; step_size_index++) {
		int maxiter_w0 = 2 * (int)(latest_optimum / step_sizes[step_size_index]);
		int maxiter_w1 = 2 * (int)(latest_optimum / step_sizes[step_size_index]);
		int maxiter_b = 2 * (int)(max*b_multiple_range / (step_sizes[step_size_index] * b_multiple));
		dim3 blocks(maxiter_w0 / 32 + 1, maxiter_w1 / 32 + 1, maxiter_b);
		dim3 threads(32, 32);
		HANDLE_ERROR(cudaMemcpy(optimal->magnitude, d_optimal_magnitude, 10000 * optimal->quantity * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(optimal->w, d_optimal_w, 10000 * optimal->quantity * optimal->size * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(optimal->b, d_optimal_b, 10000 * optimal->quantity * sizeof(float), cudaMemcpyDeviceToHost));
		optimal_best->b[0] = optimal->b[*optimal_min_index];
		optimal_best->w[0] = optimal->w[(*optimal_min_index)*optimal->size];
		optimal_best->w[1] = optimal->w[(*optimal_min_index)*optimal->size + 1];
		optimal_best->magnitude[0] = optimal->magnitude[(*optimal_min_index)];
		latest_optimum = optimal_best->magnitude[0] + step_sizes[step_size_index] * 5;
	}
}

// this CPU function is used to find the variations of w0 and w1
// for simple calculation we are only taking four combination
// this module is used to generate the array used to create variation of weights
void fillTransform(Transform* transform, int size) {
	int value[] = { 1,1,1,-1,-1,1,-1,-1 };
	transform->size = size;
	transform->value = (int*)malloc(size * pow(2, size) * sizeof(int));
	for (int i = 0; i < size * pow(2, size); i++) {
		transform->value[i] = value[i];
	}
}

//this is used to fil weights matrix with max value before iteration
void fillWeights(Weights* weights, float latest_optimum, int size) {
	weights->size = size;
	weights->value = (float*)malloc(sizeof(float) * size);
	for (int i = 0; i < size; i++) {
		weights->value[i] = latest_optimum;
	}
}

// this CPU function implements above mentioned technique
// then returns variaties of same magnitude weights matrix
void findParameterVariation(Weights* weights_transformed, Weights* weights, Transform* transform, int index) {
	weights_transformed->size = weights->size;
	weights_transformed->value = (float*)malloc(sizeof(float) * weights_transformed->size);
	for (int i = 0; i < weights_transformed->size; i++) {
		weights_transformed->value[i] = weights->value[i] * transform->value[index + i];
	}
}

// this function finds the dot product that is equavalent of the hypreplane equation
// returns the sum
float dotProduct(Weights* weights, Data* data, Data* result, float b, int index) {
	float sum = 0;
	for (int i = 0; i < data->size; i++) {
		sum += weights->value[i] * data->value[index*data->size + i];
	}
	sum += b;
	sum *= result->value[index];
	return sum;
}

// this function finds the combination of w, b with least ||w|| after one iteration over steps
// then this value will be used as new starting value of w
void findMinOptimal(Optimal* optimal, Optimal* opt_best, int index) {
	for (int i = 0; i < index; i++) {
		if (abs(opt_best->magnitude[0]) > abs(optimal->magnitude[i])) {
			opt_best->b[0] = optimal->b[i];
			opt_best->w[0] = optimal->w[i*optimal->size];
			opt_best->w[1] = optimal->w[i*optimal->size + 1];
			opt_best->magnitude[0] = abs(optimal->magnitude[i]);
		}
	}
}

//this function is not used anymore. 
// function used to sort the optimal array combinations in asending order
//this is done to get the least value combination
void sortOptimal(Optimal* optimal, int index) {
	float magnitude_temp;
	float b_temp = 0;
	float w_temp;
	for (int i = 0; i < index - 1; i++) {
		for (int j = i + 1; j < index; j++) {
			if (optimal->magnitude[i] > optimal->magnitude[j]) {
				b_temp = optimal->b[i];
				optimal->b[i] = optimal->b[j];
				optimal->b[j] = b_temp;
				magnitude_temp = abs(optimal->magnitude[i]);
				optimal->magnitude[i] = abs(optimal->magnitude[j]);
				optimal->magnitude[j] = abs(magnitude_temp);

				for (int k = 0; k < optimal->size; k++) {
					w_temp = optimal->w[i*optimal->size + k];
					optimal->w[i*optimal->size + k] = optimal->w[j*optimal->size + k];
					optimal->w[j*optimal->size + k] = w_temp;
				}
			}
		}
	}
}

// CPU function to populate sample input data
void sampleData(Data* data, Data* result) {
																			/*int data_arr[] = { 4,10,1,7,-1,12,6,12,-3,5,8,14,0,13,2,15,-5,10,10, 20,5, 15,-7,0,0,20,-5,0,-8,15,-4, 15,10, 15,1, 11,2,8,3,8,
																			5,1,6,-1,7,3,0,-5,8,0,4,0,4,-10,1,-7,-3,-6,10,-5,9,4,2,-6,-7,-13,7,-15,4, -4,13,7,15,10,12,3,10,5,-5,-9 };
																			int result_arr[] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1 };*/
	int data_arr[] = { 1,7,2,8,3,8, 5,1,7,3,6,-1 };
	int result_arr[] = { -1,-1,-1,1,1,1 };
	data->size = 2;
	result->size = 1;
	data->quantity = sizeof(data_arr) / (data->size * sizeof(int));
	result->quantity = sizeof(result_arr) / (result->size * sizeof(int));
	data->value = (int*)malloc(sizeof(int) * data->quantity * data->size);
	result->value = (int*)malloc(sizeof(int) * result->quantity * result->size);

	for (int i = 0; i < data->quantity * data->size; i++) {
		data->value[i] = data_arr[i];
	}
	for (int i = 0; i < result->quantity * result->size; i++) {
		result->value[i] = result_arr[i];
	}
}

// CPU function to populate sample predict data
void samplePredictData(Data* data) {
	int data_arr[] = { 0,10,1,3,3,4,3,5,5,5,5,6,6,-5,5,8 };
	data->size = 2;
	data->quantity = sizeof(data_arr) / (data->size * sizeof(int));
	data->value = (int*)malloc(sizeof(int) * data->quantity * data->size);

	for (int i = 0; i < data->quantity * data->size; i++) {
		data->value[i] = data_arr[i];
	}
}

// test function to check if optimal is being set or not
void sampleOptimal(Optimal* optimal) {
	int size = 2;
	float magnitude[] = { 5, 4, 3, 2, 7, 8 };
	float b[] = { 5, 4, 3, 2, 7, 8 };
	float w[] = { 5, 4, 3, 2, 7, 8 , 1, 9, 0, 5, 4, 7 };
	int quantity = 6;
	optimal->size = size;
	optimal->quantity = quantity;
	optimal->magnitude = (float*)malloc(optimal->quantity * sizeof(float));
	optimal->b = (float*)malloc(optimal->quantity * sizeof(float));
	optimal->w = (float*)malloc(optimal->quantity * sizeof(float));
	for (int i = 0; i < optimal->quantity; i++) {
		optimal->magnitude[i] = magnitude[i];
		optimal->b[i] = b[i];
		for (int j = 0; j < optimal->size; j++) {
			optimal->w[i * optimal->size + j] = w[i * optimal->size + j];
		}
	}
}

// function to find max value among the input data set
int findMax(Data* data) {
	int max = data->value[0];
	for (int i = 0; i < data->quantity*data->size; i++) {
		if (max < data->value[i]) {
			max = data->value[i];
		}
	}
	return max;
}

// function to find the min value among the input data set
int findMin(Data* data) {
	int min = data->value[0];
	for (int i = 0; i < data->quantity*data->size; i++) {
		if (min > data->value[i]) {
			min = data->value[i];
		}
	}
	return min;
}

// function to find the magnitude of the weights
float findMagnitude(Weights* weights) {
	float sum = 0;
	for (int i = 0; i < weights->size; i++) {
		sum += pow(weights->value[i], 2);
	}
	sum = sqrt(sum);
	sum = abs(sum);
	return sum;
}

// function to update the weights after an iteration
// this is a test function
void changeWeights(Weights* weights, float step) {
	for (int i = 0; i < weights->size; i++) {
		weights->value[i] -= step;
	}
}

// test function to print optimal values
void printSuspect(Optimal* optimal, int index) {
	for (int i = 0; i < index + 1; i++) {
		if (optimal->magnitude[i] < 0) {
			cout << i << ": " << optimal->magnitude[i] << ", " << optimal->w[i*optimal->size] << ", " << optimal->b[i] << " |||| ";
		}
	}
	cout << endl << " finished" << endl;
}

// CPU implementation of trainging the SVM Model
// it iterates through step, w and b to find the optimal best combination 
// of w0, w1 and b which is having the least ||W||
int fit(Data* data, Data* result, Optimal* optimal, Optimal* optimal_best, int optimal_index) {
	int max = findMax(data);
	Transform* transform = (Transform*)malloc(sizeof(Transform));
	fillTransform(transform, data->size);
	float step_sizes[] = { max*(float)0.1, max*(float)0.01, max*(float)0.001 };
	int b_multiple_range = 5;
	int b_multiple = 5;
	float latest_optimum = (float)(max * 3);
	bool optimized = false;
	bool found_option = true;
	optimal->size = data->size;
	optimal->quantity = data->quantity;
	optimal->magnitude = (float*)malloc(1000 * optimal->quantity * sizeof(float));
	//optimal->magnitude[0] = 0;
	optimal->b = (float*)malloc(10000 * optimal->quantity * sizeof(float));
	optimal->w = (float*)malloc(10000 * optimal->quantity * optimal->size * sizeof(float));
	optimal_best->size = data->size;
	optimal_best->quantity = 1;
	optimal_best->magnitude = (float*)malloc(sizeof(float));
	optimal_best->b = (float*)malloc(sizeof(float));
	optimal_best->w = (float*)malloc(optimal_best->size * sizeof(float));
	optimal_best->magnitude[0] = (float)1.5 * latest_optimum;
	optimal_best->b[0] = (float)(max * b_multiple_range);
	optimal_best->w[0] = latest_optimum;
	optimal_best->w[1] = latest_optimum;
	float equation = 0;



	for (int step_size_index = 0; step_size_index < 3; step_size_index++) {
		Weights* weights = (Weights*)malloc(sizeof(Weights));
		fillWeights(weights, latest_optimum, data->size);
		optimized = false;
		while (!optimized) {
			int limit = (int)((max*b_multiple_range) / (step_sizes[step_size_index] * b_multiple));
			int limit_index = 0;
			for (float b_index = -(float)(max*b_multiple_range); b_index < (max*b_multiple_range) && limit_index < 5 * limit; b_index += step_sizes[step_size_index] * b_multiple) {
				b_index = (float)round(b_index * 10000.0) / 10000.0;
				for (int transform_index = 0; transform_index < transform->size * pow(2, transform->size); transform_index += transform->size) {
					Weights* weights_transformed = (Weights*)malloc(sizeof(Weights));

					findParameterVariation(weights_transformed, weights, transform, transform_index);
					found_option = true;
					for (int data_index = 0; data_index < data->quantity; data_index++) {
						equation = dotProduct(weights_transformed, data, result, b_index, data_index);
						if (!(equation >= 1))
							found_option = false;
					}
					if (found_option) {
						optimal->magnitude[optimal_index] = abs(findMagnitude(weights_transformed));
						optimal->b[optimal_index] = b_index;
						for (int j = 0; j < optimal->size; j++) {
							optimal->w[optimal_index * optimal->size + j] = weights_transformed->value[j];
						}
						optimal_index++;
					}
				}
				limit_index++;
			}

			if (weights->value[0] < 0) {
				optimized = true;
			}
			else {
				changeWeights(weights, step_sizes[step_size_index]);
			}

		}
		findMinOptimal(optimal, optimal_best, optimal_index);
		latest_optimum = optimal_best->magnitude[0] + step_sizes[step_size_index] * 5;
	}
	return optimal_index;
}

// print function to display classification of the sample predict data
void printPrediction(int* result, int quantity) {
	cout << endl << "The given prediction set belong to following classes: { ";
	for (int i = 0; i < quantity; i++)
		cout << result[i] << ", ";
	cout << "}" << endl;
}

// CPU function to predict the sample result data
void predict(Data* data, Optimal* optimal) {
	float sum = 0;
	int* result = (int*)malloc(sizeof(int) * data->quantity);
	for (int j = 0; j < data->quantity; j++) {
		sum = 0;
		for (int i = 0; i < data->size; i++) {
			sum += optimal->w[i] * data->value[j*data->size + i];
		}
		sum += optimal->b[0];
		if (sum < 0)
			result[j] = -1;
		else
			result[j] = 1;
	}
	printPrediction(result, data->quantity);

}

// test function that checks the working of all the function
int test()
{
	//chrono::high_resolution_clock::time_point start, stop;
	//chrono::high_resolution_clock::time_point start_for_each_module, stop_for_each_module;
	int size = 2;
	float latest_optimum = -15;
	Transform* transform = (Transform*)malloc(sizeof(Transform));
	fillTransform(transform, size);
	cout << "Transform:";
	for (int i = 0; i < size * pow(2, size); i++) {
		cout << transform->value[i] << " , ";
	}
	cout << endl << transform->size << endl;

	Weights* weights = (Weights*)malloc(sizeof(Weights));
	fillWeights(weights, latest_optimum, size);
	cout << "Weights:";
	for (int i = 0; i < size; i++) {
		cout << weights->value[i] << " , ";
	}
	cout << endl << weights->size << endl;

	Weights* weights_transformed = (Weights*)malloc(sizeof(Weights));
	findParameterVariation(weights_transformed, weights, transform, 2);
	for (int i = 0; i < size; i++) {
		cout << weights_transformed->value[i] << " , ";
	}

	Data* data = (Data*)malloc(sizeof(Data));
	Data* result = (Data*)malloc(sizeof(Data));

	sampleData(data, result);

	cout << "Data:";
	for (int i = 0; i < data->quantity*data->size; i++) {
		cout << data->value[i] << " , ";
	}
	cout << endl << data->size << " , " << data->quantity << endl;

	cout << "Result:";
	for (int i = 0; i < result->quantity*result->size; i++) {
		cout << result->value[i] << " , ";
	}
	cout << endl << result->size << " , " << result->quantity << endl;

	cout << " DotProduct: " << dotProduct(weights, data, result, 1, 0) << endl;

	Optimal* optimal = (Optimal*)malloc(sizeof(Optimal));
	sampleOptimal(optimal);
	cout << "Optimal before sorting:";
	for (int i = 0; i < optimal->quantity; i++) {
		cout << optimal->magnitude[i] << " , " << optimal->b[i] << ",  ";
		for (int j = 0; j < optimal->size; j++) {
			cout << optimal->w[i *optimal->size + j] << " - ";
		}
		cout << "::: ";
	}
	cout << endl << optimal->size << " x " << optimal->quantity << endl;
	sortOptimal(optimal, optimal->quantity);
	cout << "Optimal after sorting:";
	for (int i = 0; i < optimal->quantity; i++) {
		cout << optimal->magnitude[i] << " , " << optimal->b[i] << ",  ";
		for (int j = 0; j < optimal->size; j++) {
			cout << optimal->w[i*optimal->size + j] << " - ";
		}
		cout << "::: ";
	}
	cout << endl << optimal->size << " x " << optimal->quantity << endl;
	cout << "Max in data: " << findMax(data) << endl;
	cout << "Min in data: " << findMin(data) << endl;
	cout << "Magnitude of weights: " << findMagnitude(weights) << endl;
	changeWeights(weights_transformed, latest_optimum);
	cout << " Step: " << latest_optimum << endl;
	cout << "Updated weights: " << endl;
	for (int i = 0; i < size; i++) {
		cout << weights_transformed->value[i] << " , ";
	}
	cout << endl;
	cout << endl;
	cout << endl;

	Optimal* optimal_best = (Optimal*)malloc(sizeof(Optimal));
	int optimal_index = 0;
	optimal_index = fit(data, result, optimal, optimal_best, optimal_index);
	cout << "magnitude: " << optimal_best->magnitude[0] << endl;
	cout << "W: " << optimal_best->w[0] << " x " << optimal_best->w[1] << endl;
	cout << "B: " << optimal_best->b[0];

	Data* predictData = (Data*)malloc(sizeof(Data));
	samplePredictData(predictData);
	cout << endl << "Data to predicted:";
	for (int i = 0; i < predictData->quantity*predictData->size; i++) {
		cout << predictData->value[i] << " , ";
	}
	cout << endl << predictData->size << " , " << predictData->quantity << endl;
	predict(predictData, optimal_best);
	return 0;
}

int main() {
	// chrno code is commented as it was not working in xsede
	//chrono::high_resolution_clock::time_point start, stop;
	//chrono::high_resolution_clock::time_point start_for_each_module, stop_for_each_module;

	Data* data = (Data*)malloc(sizeof(Data));
	Data* result = (Data*)malloc(sizeof(Data));
	//start = chrono::high_resolution_clock::now();
	//start_for_each_module = chrono::high_resolution_clock::now();
	sampleData(data, result);
	//stop_for_each_module = chrono::high_resolution_clock::now();
	//chrono::milliseconds duration_sampleData = chrono::duration_cast<chrono::milliseconds>(stop_for_each_module - start_for_each_module);


	Optimal* optimal = (Optimal*)malloc(sizeof(Optimal));
	Optimal* optimal_best = (Optimal*)malloc(sizeof(Optimal));
	//start_for_each_module = chrono::high_resolution_clock::now();
	int optimal_index = 0;
	optimal_index = fit(data, result, optimal, optimal_best, optimal_index);
	//stop_for_each_module = chrono::high_resolution_clock::now();
	//chrono::milliseconds duration_fit = chrono::duration_cast<chrono::milliseconds>(stop_for_each_module - start_for_each_module);


	Data* predictData = (Data*)malloc(sizeof(Data));
	//start_for_each_module = chrono::high_resolution_clock::now();
	samplePredictData(predictData);
	//stop_for_each_module = chrono::high_resolution_clock::now();
	//chrono::milliseconds duration_samplePredict = chrono::duration_cast<chrono::milliseconds>(stop_for_each_module - start_for_each_module);


	//start_for_each_module = chrono::high_resolution_clock::now();
	predict(predictData, optimal_best);
	//	stop_for_each_module = chrono::high_resolution_clock::now();
		//chrono::milliseconds duration_predict = chrono::duration_cast<chrono::milliseconds>(stop_for_each_module - start_for_each_module);


		//stop = chrono::high_resolution_clock::now();
		//chrono::milliseconds duration_cpu = chrono::duration_cast<chrono::milliseconds>(stop - start);


	cout << endl << "      ******************************      " << endl << endl;
	/*cout << "Duration for filling sample data and result: " << duration_sampleData.count()<<endl;
	cout << "Duration for fitting data into model: " << duration_fit.count() << endl;
	cout << "Duration for filling sample prediction data: " << duration_samplePredict.count() << endl;
	cout << "Duration for testing model with predicting data: " << duration_predict.count() << endl;
	cout << "CPU execution time for SVM modelling and prediction: " << duration_cpu.count();*/
	cout << endl << endl << "      ******************************      " << endl;
	Optimal* d_optimal = (Optimal*)malloc(sizeof(Optimal));
	Optimal* d_optimal_best = (Optimal*)malloc(sizeof(Optimal));
	cout << "Optimal b: " << optimal_best->b[0] << endl;
	cout << "Optimal weights: " << optimal_best->w[0] << " x " << optimal_best->w[1] << endl;
	cout << "Optimal weights magnitude: "<< optimal_best->magnitude[0] << endl;
	//d_fit(data, result, d_optimal, d_optimal_best);
	//cout << optimal_best->b[0] << endl;
	//cout << optimal_best->w[0] << " x " << optimal_best->w[1] << endl;
	//cout << optimal_best->magnitude[0] << endl;
	//cout << optimal_best->quantity << endl;
	//cout << optimal_best->size << endl;
	d_findMinOptimal(optimal, d_optimal_best, optimal_index);	
	d_callpredict(predictData, optimal_best);
return 0;
}