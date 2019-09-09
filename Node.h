#pragma once

#define EIGEN_USE_THREADS 

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <vector>
#include <map>
#include <memory>
#include "Op.h"

//typedef Eigen::MatrixXf NodeBase;
typedef int  NodeBase;

using Eigen::MatrixXf;
using Eigen::ArrayXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::Tensor;
using Eigen::array;

template<typename T>

class Node{

	public:
		T value;
		T gradient;
		opType op;
		std::vector <std::shared_ptr<Node>> children;	
		std::vector <Node*> parents;
		std::vector <int> parentType;
		std::vector <int> childType;
		std::map <Node*, T> gradientParent;
		bool isRoot = false;
		bool variable;
		std::vector<int> valueDims;
	
		// Op Specific Cache
		Tensor<float, 4> flattenGradient;
		Tensor<Eigen::Index, 4> argMaxCache;
		T gradCache;
	
		Node(const T& value);
		Node(const T& value, bool variable);
		Node();
		
		//T& eval();
		void im2col(T& input, T& kernel, Tensor<float, 2>& inputMat, Tensor<float, 2>& kernelRow);
		void im2colImage(T& input, Tensor<float, 2>& inputMat, int kernelSize);
		void col2im(T& input, T& kernel, Tensor<float, 2>& inputMat, Tensor<float, 2>& kernelRow);
		void col2imImage(T& input, Tensor<float, 2>& inputMat, int kernelSize);
		void eval();
		void back();
		bool areChildrenDone();
		void update();
	
	private:
		void setValue(Node& var);
};


template <typename T>
Node<T>& maxPool2D(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_maxpool, nodesVector);
}

template <typename T>
Node<T>& softmax(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_softmax, nodesVector);
}

template <typename T>
Node<T>& operator -(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_minus, nodesVector);
}

template <typename T>
Node<T>& operator +(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_plus, nodesVector);
}

template <typename T>
Node<T>& operator *(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_mul, nodesVector);
}

template <typename T>
Node<T>& emul(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_emul, nodesVector);
}

template <typename T>
Node<T>& max(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_max, nodesVector);
}

template <typename T>
Node<T>& colmax(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_colmax, nodesVector);
}

template <typename T>
Node<T>& conv(Node<T>& first, Node<T>& second){
	std::vector<Node<T>*> nodesVector = {&first, &second};
	return store(opType::e_conv, nodesVector);
}

template <typename T>
Node<T>& relu(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_relu, nodesVector);
}

template <typename T>
Node<T>& exp(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_exp, nodesVector);
}

template <typename T>
Node<T>& sum(Node<T>& first, int dim){
	std::vector<Node<T>*> nodesVector = {&first};
	if(dim == 0){return store(opType::e_rowsum, nodesVector);}
	else if(dim == 1){return store(opType::e_colsum, nodesVector);}
	else if(dim == -1){return store(opType::e_sum, nodesVector);}
	else { std::cout << "Wrong dimension" << std::endl; Node<T>* newNode = new Node<T>(); return *newNode;}
}

template <typename T>
Node<T>& nlog(Node<T>& first){
	std::vector<Node<T>*> nodesVector = {&first};
	return store(opType::e_log, nodesVector);
}

template <typename T>
Node<T>& store(opType op, std::vector<Node<T>*>& nodesVector){

	std::shared_ptr<Node<T>> newNode = std::shared_ptr<Node<T>>(new Node<T>());	
	//Node<T>* newNode = new Node<T>();
	newNode->op = op;
	newNode->variable = false;

	// Store both parents to node as shared pointers
	for(Node<T>* n: nodesVector){
		//std::cout << "Parents are: " << n << std::endl;
		newNode->parentType.push_back(0);
		newNode->parents.push_back(n);
		n->children.push_back(newNode);
		n->childType.push_back(0);
	}
	//std::cout << "Here" << std::endl;	
	//std::cout << "Newnode parents: " << newNode->parents.size() << std::endl;
	return *newNode;
}



Node<MatrixXf>& storeFlatten(opType op, std::vector<Node<Tensor<float, 4>>*>& nodesVector);

Node<MatrixXf>& flatten(Node<Tensor<float, 4>>& first);
