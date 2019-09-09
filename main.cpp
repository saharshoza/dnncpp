#include <iostream>
#include <initializer_list>
#include <fstream>
#include <random>
#include "Graph.h"

#include <CL/cl.h>

// Input
#define WIDTH 28
#define HEIGHT 28
#define DEPTH 1
#define EXAMPLES 50
#define CLASSES 10

// Filter Dimensions
#define KERNELSIZE 4
#define NUMFILTERS 32
#define STRIDE 1


// Maxpool Dimensions
#define MAXPOOL_STRIDE 1

#define CEILING(x,y) (((x) + (y) - 1) / (y))

// Flatten Features
#define FEATSIZE CEILING((WIDTH-KERNELSIZE+1),2)*CEILING((HEIGHT-KERNELSIZE+1),2)*NUMFILTERS

void loadLabel(const std::string &Yfile, MatrixXf& YMat){

	std::ifstream yFile(Yfile);
	std::string yLine;
	int YNum = 0;
	
	while(YNum < EXAMPLES){
		std::getline(yFile, yLine);
		std::istringstream yiss(yLine);

		std::string valString;

		for(int c = 0; c < CLASSES; ++c){
			std::getline(yiss, valString, ' ');
			//std::cout << valString << std::endl;
			float yFloat = std::stod(valString);
			YMat(YNum, c) = yFloat;
		}
		
		++YNum;
		
	}
	
	std::cout << "Labels" << std::endl;
	std::cout << YMat << std::endl;

	return;
}


void loadData(const std::string &Xfile, Tensor<float, 4>& XTensor){
	
	std::ifstream xFile(Xfile);
	std::string xLine, yLine;
	int XNum = 0;
	
	while(XNum < EXAMPLES){
		
		std::getline(xFile, xLine);
		std::istringstream xiss(xLine);

		std::string valString;

		for(int d = 0; d < DEPTH; ++d){
			for(int h = 0; h < HEIGHT; ++h){
				for(int w = 0; w < WIDTH; ++w){

					std::getline(xiss, valString, ' ');
					float valFloat = std::stod(valString);
					XTensor(h, w, d, XNum) = valFloat;

				}
			}
		}

		std::cout << "Image " << XNum << std::endl;
		array<long, 4> start = {0, 0, 0, XNum};
		array<long, 4> end = {HEIGHT, WIDTH, DEPTH, 1};
		std::cout << XTensor.slice(start, end) << std::endl;
		++XNum;
	}


	Tensor<float, 1>::Dimensions normalizeDim{3};
	Tensor<float, 3> meanTensor = XTensor.mean(normalizeDim);
	Eigen::array<long, 4> reshapeDim({HEIGHT, WIDTH, DEPTH, 1});
	Tensor<float, 4> meanTensor4D = meanTensor.reshape(reshapeDim);
	Eigen::array<int, 4> bcast({1, 1, 1, EXAMPLES});
	Tensor<float, 4> meanTensor4DBcast = meanTensor4D.broadcast(bcast);

	std::cout << meanTensor4D << std::endl;

	XTensor = XTensor - meanTensor4DBcast;
	XTensor = XTensor/255.0f;
	
	std::cout << "Normalized X " << std::endl;
	array<long, 4> start = {0, 0, 0, 0};
	array<long, 4> end = {HEIGHT, WIDTH, DEPTH, 1};
	std::cout << XTensor.slice(start, end) << std::endl;
	
	return;
}


int main(){

	cl_int err;
	cl_uint numPlatforms;

	err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS == err)
         printf("\nDetected OpenCL platforms: %d", numPlatforms);
    else
         printf("\nError calling clGetPlatformIDs. Error code: %d", err);

	exit(0);

	Graph g;

	// Input Features

	Tensor<float, 4> XTensor(WIDTH, HEIGHT, DEPTH, EXAMPLES);
	loadData("data/mnist/mnistX.out", XTensor);
	Node<Tensor<float, 4>> XTNode(XTensor, false);

	// Labels

	MatrixXf YMat = MatrixXf::Zero(EXAMPLES, CLASSES);
	loadLabel("data/mnist/mnistY.out", YMat);
	//for(int i = 0; i < EXAMPLES; ++i){YMat(i, i%CLASSES) = 1;}
	Node<MatrixXf> Y(YMat, false);

	// Parameters

	// Conv Filter Parameters

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-0.5, 0.5);
	//std::normal_distribution<double> distribution(0.0, 3.0);

	Tensor<float, 4> FTensor(KERNELSIZE, KERNELSIZE, DEPTH, NUMFILTERS);
	FTensor.setConstant(0.0f);
	for(int f = 0; f < KERNELSIZE; ++f){
		for(int f2 = 0; f2 < KERNELSIZE; ++f2){
			for(int d = 0; d < DEPTH; ++d){
				for(int n = 0; n < NUMFILTERS; ++n){
					FTensor(f, f2, d, n) = distribution(generator);
				}
			}
		}
	}
	//FTensor.setRandom<Eigen::internal::UniformRandomGenerator>();	
	Node<Tensor<float, 4>> FTNode(FTensor);

	// Affine Parameters
	Node<MatrixXf> W(MatrixXf::Random(FEATSIZE, CLASSES));
	Node<MatrixXf> B(MatrixXf::Random(1, CLASSES));

	// Conv - Relu - Maxpool - Flatten

	Node<Tensor<float, 4>>& Co = conv(XTNode, FTNode);
	Node<Tensor<float, 4>>& CoRel = relu(Co);
	Node<Tensor<float, 4>>& CoMax = maxPool2D(CoRel);
	//Node<Tensor<float, 4>>& CoMax = maxPool2D(Co);
	Node<MatrixXf>& CoFlat = flatten(CoMax);
	Node<MatrixXf>& A = CoFlat * W; 		

	// Softmax
	Node<MatrixXf>& Z0 = A + B;
	Node<MatrixXf>& Z = relu(Z0);
	//Z.value = Z.value - Z.value.maxCoeff();
	Node<MatrixXf>& softmax = nlog(sum(exp(Z), 1)) - Z;
	Node<MatrixXf>& loss = sum(emul(Y, softmax), -1);

	clock_t startTime;

	for(int iter = 0; iter < 1000; iter++){

		startTime = clock();
		g.eval(loss);
		std::cout << "Forward Pass " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << std::endl;
		std::cout << "loss " << loss.value << std::endl;

/*

		std::cout << "FTNode value " << FTNode.value << std::endl;
		std::cout << "Co value " << Co.value << std::endl;
		std::cout << "CoMax value " << CoMax.value << std::endl;
		std::cout << "CoFlat value " << CoFlat.value << std::endl;
		std::cout << "W value " << W.value << std::endl;
		std::cout << "A value " << A.value << std::endl;
		std::cout << "Z0 value " << Z0.value << std::endl;
		std::cout << "Z value " << Z.value << std::endl;
		std::cout << "softmax value " << softmax.value << std::endl;
		std::cout << "Y value " << Y.value << std::endl;
*/
		
		startTime = clock();
		g.backProp(loss);
		std::cout << "Backward Pass " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << std::endl;

/*
		std::cout << "FTNode " << FTNode.gradient << std::endl;
		std::cout << "Co " << Co.gradient << std::endl;
		std::cout << "CoMax " << CoMax.gradient << std::endl;
*/
		startTime = clock();
		g.update(loss);
		std::cout << "Gradient update " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << std::endl;

	}

	std::cout << "Filter" << std::endl;
	std::cout << FTNode.value << std::endl;
	std::cout << "W" << std::endl;
	std::cout << W.value << std::endl;
	std::cout << "B" << std::endl;
	std::cout << B.value << std::endl;
}
