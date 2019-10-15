# svm-using-cuda-without-libraries
SVM without using libraries in CUDA

## Description
Support Vector Machines are supervised learning models for classification and regression problems. They can solve linear and non-linear problems and work well for many practical problems [1]. The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space (N — the number of features) that distinctly classifies the data points [2]. Through this project, I plan to implement the SVM in CPU and GPU using CUDA. To implement it, the focus will be to calculate a hyperplane that separates sample data set into classes to find the least parameters. Using those parameters, we will be able to find classes for new data. By implementing in GPU, we can obtain higher performance over CPU implementation. To measure the performance difference, Nvidia Profiler is going to be used.

## Theory
The idea of SVM is to implement a hyper plane that lies at maximum distance from all data points. The distance of a data point from hyperplane is the vector projection. This projection needs to maximize for all data points. This is calculated as parameter, w for the model. However, some data points may fall on the other side of the hyperplane even if it belongs to other class. To adjust this, we need to add another factor called bias, b.  The best model will have low w and low b which means that all data points have large margin from hyper plane and lowest bias to include all data points. Hyper plane is defined as:
```
y ̅_i (x ̅_i⋅w ̅+b)-1=0
Where y ̅_i - class of a data point
x ̅_i - input data point (features)
w ̅ - weights
b - bias
```
There are a lot of possible hyperplanes that satisfy the above equation as shown below. Out of these number of hyperplanes, one of them will have least magnitude for w and b. The idea for finding optimal w and b is an optimization problem which is implemented through this project in both CPU and GPU.

The idea of optimization is as shown below where x axis shows difference of y ̅_i (x ̅_i⋅w ̅+b)-1 from 0 and y axis shows magnitude of w. As shown in the document, first jump will be using one learning step value and then step value will be decreased as it gets closer and closer to median value. Once the value are calculated we can classify other data points using the value of y ̅_i (x ̅_i⋅w ̅+b)-1. If this is greater than 0 then it belongs to class 1. Otherwise it belongs to class -1.

## Implementation
### CPU
A high level overview of the implementation in CPU is as follows:
1.	Find maximum value in the input data set
2.	Initialize weights vector w with max value. i.e w0 and w1 is equal to max value
3.	Define max value for b
4.	Define step sizes
5.	Iterate though step sizes.
6.	For each step size, Iterate over b from -(max value * 5) to (max value *5) with intervals at step size*5
7.	Along with this iteration, iterate over w from –(max value * 10) to (max value * 10) with intervals at step size.
8.	Thought the magnitude of w will be same, there can be many cases of w that can happen. So we find out (w0,w1), (w0,-w1), (-w0,-w1), and (-w0,w1)
9.	For all these combinations find if it satisfies hyperplane equation.
10.	If yes, then store the combination until one iteration over step is over.
11.	When that iteration is over, then find least value of optimal.
12.	This value is store as the value of w.
13.	Continue till all iterations are over.
14.	Find least magnitude index of w, and store w0, w1 and b of that combination.
15.	Use these parameters to predict the new data set.

### GPU
Implementation in GPU is far faster owing to the parallel execution. Considering this advantage, the implementation is divided into three kernels:
#### d_getAllOptimal
This kernel function used to find all possible combination of ||w||, w & b. It is done by scanning through a 3d matrix given by weights and bias. X axis is denoted by w0, y axis is denoted by w1 and z axis by b. The values in each axis represents the number of iterations/divisions within the range. w0 and w1 ranges from w_min - w_max with division = w range/step value. i.e w0 and w1 is equal to 2 * (maximum value of input data) * 10  / step, b is equal to (maximum value of input data * 5)/ step value for the first iteration. These dimensions are used to decide number of blocks and threads. This is shown below:
```
<<< (#of w division/32 +1, #of w1 divisions/32, # of b divisions), (32,32) >>>
```
The result of this kernel is a list of combination of ||w||, w0, w1, and b. This list is then searched for minimum value.

#### d_findMinOptimal 
This module calls cublas functions to search through the list of magnitudes ||w|| to find the index of the least magnitude. Then cublas returns the index which is then used to find the optimal b and optimal w0 and w1. cublasIsamin is the function used to find the index of the lowest magnitude. Syntax of the cublasIsamin is shown below:
```
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x, int incx, int *result)
```
The result of this kernel is a set of w0, w1, b and magnitude that is used later to predict the value.

#### 	d_callpredict 
Executes the hyperplane equation for data set for prediction. This kernel received the set with least magnitude, w0, w1 and b. It uses the hyper plane equation shown below to calculate the class.
```
hyper plane equaltion=(x ̅_i⋅w ̅+b)-1
```
If this equation is less than 0, then class is -1. Otherwise class is 1. This kernel is executed with the following dimension
```
<<< (# of data points/32 +1, size of data points/32 + 1), (32, 32)>>>
```
The resultant array contains the classification for each data points.

## Execution
The output statments are printed ot a.out file.

## Profiling
Using high precision timer, CPU implementation was timed to find the results as follows:
```
Duration for fitting sample data and result: 0
Duration for fitting data into model: 1295 ms
Duration for testing model with prediction data: 4
Total CPU Exectuiong time for SVM modelling and prediction: 1299 ms
```
GPU implementation was profiled using Nvidia Visual Profiler as shown below:
```
Execution Time for d_findMinOptimal: 9.118 micro sec
Importance of d_findMinOptimal: 75.7%
Execution Time for d_callPredict:2.933 miro sec
Importance of d_callPredict: 24.3%
Total Execution Time in GPU: 22.176 micro sec
```
Additional profiling results are included in the document.

## Continuing Problems/Future Improvements
1. cublasIsamin is always ignoring the last one fifth of portion o f the set of all possible optimal values. It is returning a value of ||w|| that is not the least element. As a work around I used the optimal value from CPU Implementation.
2. 

## Conclusion
GPU implementation is so much faster than CPU. GPU completes simulation in micro seconds while CPU completes in milliseconds. Profiling on kernels gave idea on how code performs and how it can be improved. This project also gave an overview of how fast this algorithm works on GPU and how it can be used for regression as well as multi class classification problem.
