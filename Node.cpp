#include <iostream>
#include "Node.h"


template <typename T>
Node<T>::Node(const T& value):
        value(value),
        isRoot(true),
        op(opType::e_nop),
        variable(true){}


template <typename T>
Node<T>::Node(const T& value, bool variable):
        value(value),
        isRoot(true),
        op(opType::e_nop),
        variable(variable){}


template <typename T>
Node<T>::Node(){};


template <>
//MatrixXf& Node<MatrixXf>::eval(){
void Node<MatrixXf>::eval(){
        gradient = MatrixXf::Zero(0,0);
        switch(op){
                case opType::e_flatten:{
                        Node<MatrixXf>* parentNodeMatrix = parents[0];
                        Node<Tensor<float, 4>>* parentNode = reinterpret_cast<Node<Tensor<float, 4>>*>(parentNodeMatrix);
                        float* data =  parentNode->value.data();
                        const Tensor<float, 4>::Dimensions& parentDim = parentNode->value.dimensions();
                        MatrixXf flattenedMat(parentDim[3], parentDim[0]*parentDim[1]*parentDim[2]);
                        for(int num = 0; num < parentDim[3]; ++num){
                                array<long, 4> start = {0, 0, 0, num};
                                array<long, 4> extent = {parentDim[0], parentDim[1], parentDim[2], 1};
                                array<Eigen::DenseIndex, 1> one_dim({parentDim[0] * parentDim[1] * parentDim[2] });
                                Tensor<float, 1> features = parentNode->value.slice(start, extent).reshape(one_dim);
                                MatrixXf featuresMat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(features.data(), 1, parentDim[0]*parentDim[1]*parentDim[2]);
                                flattenedMat.row(num) = featuresMat;
                                
                        }
/*
                        std::cout << "parentNode" << std::endl;
                        std::cout << parentNode->value << std::endl;
                        std::cout << "parentNode example 0" << std::endl;
                        std::cout << parentNode->value.chip(0,3) << std::endl;
                        std::cout << "parentNode example 0 dim 3" << std::endl;
                        std::cout << parentNode->value.chip(0,3).chip(0,2) << std::endl;
                        std::cout << "parentNode example 0 dim 3" << std::endl;
                        std::cout << parentNode->value.chip(0,3).chip(1,2) << std::endl;
                        std::cout << "Flattened" << std::endl;
                        std::cout << flattenedMat << std::endl;
                        
                        std::cout << "Flattened example 0" << std::endl;
                        std::cout << flattenedMat.row(0) << std::endl;
                        MatrixXf exampleOne = flattenedMat.row(0);
                        Eigen::TensorMap<Tensor<float, 3>> tmp(exampleOne.data(), parentDim[0], parentDim[1], parentDim[2]);
                        std::cout << "Reconstructed example 0" << std::endl;
                        std::cout << tmp << std::endl;
*/
                        value = flattenedMat;
                        return;
                }
                case opType::e_plus:{


                        MatrixXf v0 = parents[0]->value;
                        MatrixXf v1 = parents[1]->value;
                        
                        //std::cout << v0Mat << std::endl;
                        //std::cout << v1Mat << std::endl;
                        if(v0.rows() == v1.rows() && v0.cols() == v1.cols())
                                {value = v0+v1;}
                        else if(v0.rows() == 1){value = v1.rowwise() + v0.row(0);}
                        else if(v1.rows() == 1){value = v0.rowwise() + v1.row(0); }
                        else {std::cout << "Invalid addition" << std::endl; exit(1);}
                        return;
                }
                case opType::e_minus:{
                        MatrixXf v0 = parents[0]->value;
                        MatrixXf v1 = -1*parents[1]->value;
                        //std::cout << v0 << std::endl;
                        //std::cout << v1 << std::endl;
                        if(v0.rows() == v1.rows() && v0.cols() == v1.cols())
                                {value = v0+v1;}
                        else if(v0.rows() == 1){value = v1.rowwise() + v0.row(0);}
                        else if(v1.rows() == 1){value = v0.rowwise() + v1.row(0); }
                        else if(v0.cols() == 1){value = v1.colwise() + v0.col(0); }
                        else {std::cout << "Invalid subtraction" << std::endl; exit(1);}
                        return;
                }
                case opType::e_mul:{
                        value = parents[0]->value * parents[1]->value;
                        return;
                }
                case opType::e_emul:{
                        value = parents[0]->value.cwiseProduct(parents[1]->value);
                        return;
                }
                case opType::e_colsum:{
                        MatrixXf sumMat(1,1);
                        sumMat << parents[0]->value.sum();
                        value = parents[0]->value.rowwise().sum();
                        return;
                }
                case opType::e_rowsum:{
                        MatrixXf sumMat(1,1);
                        sumMat << parents[0]->value.sum();
                        value = parents[0]->value.colwise().sum();
                        return;
                }
                case opType::e_sum:{
                        MatrixXf sumMat(1,1);
                        sumMat << parents[0]->value.sum();
                        value = sumMat;
                        return;
                }
                case opType::e_exp:{
                        value = parents[0]->value.array().exp().matrix();
                        return;
                }
                case opType::e_log:{
                        value = parents[0]->value.array().log().matrix();
                        return;
                }
                case opType::e_max:{
                        if(parents[1]->value.size() == 1){ 
                                value = parents[0]->value.cwiseMax(parents[1]->value(0,0));}
                        else{value = parents[0]->value.cwiseMax(parents[1]->value);}
                        return;
                }
                case opType::e_relu:{
                        value = parents[0]->value.cwiseMax(0);
                }        
                case opType::e_nop: {
                        return;
                }
                default:{
                        std::cout << "Unknown Op" << std::endl;
                        return;
                }
        }
}




template <>
//Tensor<float, 4>& Node<Tensor<float, 4>>::eval(){
void Node<Tensor<float, 4>>::eval(){
        gradient.setZero();
        switch(op){
                case opType::e_conv:{
                        parentType[0] = 1;
                        parentType[1] = 1;
                        array<int, 3> dims = {0, 1, 2};
                        const Tensor<float, 4>::Dimensions& kernelDim = parents[1]->value.dimensions();
                        int kernelSize = kernelDim[0];
                        int numFilters = kernelDim[3];
                        const Tensor<float, 4>::Dimensions& inputDim = parents[0]->value.dimensions();
                        int height = inputDim[0];
                        int width = inputDim[1];
                        int depth = inputDim[2];
                        int numExamples = inputDim[3];
                        Tensor<float, 4> out(height-kernelSize+1, width-kernelSize+1, numFilters, numExamples);
                        out.setConstant(0.0f);
                        for(int n = 0; n < numExamples; ++n){ 
                                for(int f = 0; f < numFilters; ++f){
                                        array<long, 4> inputStart = {0, 0, 0, n};
                                        array<long, 4> inputExtent = {height, width, depth, 1};
                                        array<long, 4> outStart = {0, 0, f, n};
                                        array<long, 4> outExtent = {height-kernelSize+1, width-kernelSize+1, 1, 1};
                                        array<long, 4> kernelStart = {0, 0, 0, f};
                                        array<long, 4> kernelExtent = {kernelSize, kernelSize, depth, 1};
                                        Tensor<float, 4> currKernel = parents[1]->value.slice(kernelStart, kernelExtent);
                                        Tensor<float, 4> currInput  = parents[0]->value.slice(inputStart, inputExtent);
                                        out.slice(outStart, outExtent) = currInput.convolve(currKernel, dims);
                                }
                        }
                        value = out;
                        parents[0]->childType[0] = 1;
                        parents[1]->childType[0] = 1;
                        return;        
                }
                case opType::e_maxpool: {
                        parentType[0] = 1;
                        int poolSize = 2;
                        int poolStride = 1;
                        Tensor<float, 4> v = parents[0]->value;        
                        const Tensor<float, 4>::Dimensions& inputDim = parents[0]->value.dimensions();
                        int numStrides0 = inputDim[0]/poolSize+1;
                        int numStrides1 = inputDim[1]/poolSize+1;
                        Tensor<float, 4> out((inputDim[0]+1)/poolSize, (inputDim[1]+1)/poolSize, inputDim[2], inputDim[3]);
                        Tensor<Eigen::Index, 4> arg((inputDim[0]+1)/poolSize, (inputDim[1]+1)/poolSize, inputDim[2], inputDim[3]);
                        
                        for(int n = 0; n < inputDim[3]; ++n){ 
                                for(int f = 0; f < inputDim[2]; ++f){
                                        //for(int p0 = 0; p0 < numStrides0; p0+=poolSize){
                                        for(int p0 = 0; p0 < numStrides0; p0+=poolStride){
                                                //for(int p1 = 0; p1 < numStrides1; p1+=poolSize){
                                                for(int p1 = 0; p1 < numStrides1; p1+=poolStride){
                                                        array<long, 4> inputStart = {p0, p1, f, n};
                                                        array<long, 4> inputExtent = {poolSize, poolSize, 1, 1};
                                                        Tensor<float, 4> inputSlice = v.slice(inputStart, inputExtent);
                                                        //std::cout << inputSlice.maximum() << std::endl;
                                                        Tensor<float, 0> tmp =  inputSlice.maximum();
                                                        Tensor<Eigen::Index, 0> argIndex =  inputSlice.argmax();
                                                        //std::cout << "Argmax is " <<  inputSlice.argmax() << std::endl;;
                                                        //std::cout << "InputSlice " << inputSlice << std::endl;
                                                        //out(p0/poolSize, p1/poolSize, f, n) = *tmp.data();
                                                        //arg(p0/poolSize, p1/poolSize, f, n) = *argIndex.data();
                                                        out(p0, p1, f, n) = *tmp.data();
                                                        arg(p0, p1, f, n) = *argIndex.data();
                                                        }
                                                }
                                        }
                                }
                        value = out;
                        argMaxCache = arg;
                        parents[0]->childType[0] = 1;
                        return;        
                }
                case opType::e_nop: {
                        return;
                }
                case opType::e_relu:{
                        parentType[0] = 1;
                        Tensor<float, 4> zeroMat(parents[0]->value.dimensions());
                        zeroMat.setConstant(0);
                        value = parents[0]->value.cwiseMax(zeroMat);
                        parents[0]->childType[0] = 1;
                        return;
                }
                default:{
                        std::cout << "Unsupported type " << op << std::endl;
                        exit(0);
                }
        }
}


template <typename T>
void Node<T>::eval(){
        return ;
}


template <>
bool Node<NodeBase>::areChildrenDone(){
        return false;
}


template <typename T>
bool Node<T>:: areChildrenDone(){
        //std::cout << "Entering common areChildrenDone " <<  std::endl;
        int childId = 0;
        for(auto n: children){
                //std::cout << "Child Type is " << childType[childId] << std::endl;
                if(childType[childId]){
                        if(n->gradientParent.find(this) == n->gradientParent.end()){ 
                                std::cout << "Unable to find in child " << n.get() << std::endl;
                                return false;}
                        else{
                                //std::cout << "Gradient size "  << gradient.size() << std::endl;
                                if(gradient.size() == 0){ 
                                        //std::cout << "Setting gradient" << std::endl;
                                        gradient = n->gradientParent[this];}
                                else{        //std::cout << "Adding here" << std::endl;
                                        gradient += n->gradientParent[this];
                                        //std::cout << "Added gradient is " << gradient << std::endl;
                                }
                        }
                }
                else{
                        Node<MatrixXf>* nodeMat = reinterpret_cast<Node<MatrixXf>*> (n.get());
                        Node<MatrixXf>* selfReinterpret = reinterpret_cast<Node<MatrixXf>*> (this);
                        //std::cout << "Reintepreting myself " << std::endl;
                        if(nodeMat->gradientParent.find(selfReinterpret) == nodeMat->gradientParent.end()){ 
                                //std::cout << "Unable to find in child " << n.get() << std::endl;
                                return false;}
                        else{
                                if(nodeMat->op != opType::e_flatten){
                                        //std::cout << "areChildrenDone got MatrixXf child but op not e_flatten " << std::endl;
                                        exit(0);
                                }
                                else{ T* tmp = reinterpret_cast<T*> (&nodeMat->flattenGradient); gradient = *tmp;}
                        }
                }
                childId++;
        }
        return true;
}


template <>
bool Node<MatrixXf>:: areChildrenDone(){
        //std::cout << "Entering MatrixXf areChildrenDone " << std::endl;
        //std::cout << "Number of children " << children.size() << std::endl;
        for(auto n: children){
                if(n->gradientParent.find(this) == n->gradientParent.end()){ 
                        //std::cout << "Unable to find in child " << n.get() << std::endl;
                        return false;}
                else{
                        //std::cout << "Gradient size "  << gradient.size() << std::endl;
                        if(gradient.size() == 0){ gradient = n->gradientParent[this];}
                        else{        gradient += n->gradientParent[this]; }
                }


        }
        return true;
}
template <typename T>
void Node<T>::back(){
        return;
}


template <>
void Node<Tensor<float, 4>>::col2im(Tensor<float, 4>& input, Tensor<float, 4>& kernel, Tensor<float, 2>& flattenGradient, Tensor<float, 2>& kernelRow){ 


        const Tensor<float, 2>::Dimensions& flattenDim = flattenGradient.dimensions();
        const Tensor<float, 4>::Dimensions& kernelDim = kernel.dimensions();
        const Tensor<float, 4>::Dimensions& inputDim = input.dimensions();
/*
        std::cout << "flattenDim " << flattenDim[0] << " " << flattenDim[1] << " " << flattenDim[2] << std::endl;
        std::cout << "kernelDim " << kernelDim[0] << " " << kernelDim[1] << " " << kernelDim[2] << " " << kernelDim[3] << std::endl;
        std::cout << "inputDim " << inputDim[0] << " " << inputDim[1] << " " << inputDim[2] << " " << inputDim[3] <<std::endl;
*/
        int stride = 1;
        int hPrime = (inputDim[0] - kernelDim[0])/stride + 1;        
        int wPrime = (inputDim[1] - kernelDim[1])/stride + 1;        
/*
        std::cout << "hPrime " << hPrime << std::endl;
        std::cout << "wPrime " << wPrime << std::endl;
*/


        for(int i = 0; i < flattenDim[0]; ++i){
                Tensor<float, 1> row = flattenGradient.chip(i, 0);
                //std::cout << "Row " << std::endl;
                //std::cout << row << std::endl;
                Eigen::array<long, 4> reshapeDim({kernelDim[0], kernelDim[1], inputDim[2], 1});
                Tensor<float, 4> rowReshape = row.reshape(reshapeDim);
                //std::cout << rowReshape << std::endl;
                int hStart = (i / wPrime) * stride;
                int wStart = (i % wPrime) * stride;
                Eigen::array<long, 4> startExtent = {hStart, wStart, 0, 0};
                Eigen::array<long, 4> endExtent = {kernelDim[0], kernelDim[1], inputDim[2], 1};
                input.slice(startExtent, endExtent) += rowReshape;        
                //std::cout << "col2im conversion " << std::endl;
                //std::cout << input << std::endl;
        }
        //exit(0);


        Eigen::array<long, 4> kernelReshapeDim({kernelDim[0], kernelDim[1], kernelDim[2], 1});
        kernel = kernelRow.reshape(kernelReshapeDim);


        return;
}


template <>
void Node<Tensor<float, 4>>::im2col(Tensor<float, 4>& input, Tensor<float, 4>& kernel, Tensor<float, 2>& outputMat, Tensor<float, 2>& kernelRow){ 


        const Tensor<float, 4>::Dimensions& inputDim = input.dimensions();
        const Tensor<float, 4>::Dimensions& kernelDim = kernel.dimensions();
        int kernelSize = kernelDim[0];
        int numFilters = kernelDim[3];
        //std::cout << inputDim[0] << " " << inputDim[1] << " " << inputDim[2] << " " << inputDim[3] << std::endl;
        int stride = 1;
        const Tensor<float, 2>::Dimensions& outputMatDim = outputMat.dimensions();
        int newHeight = (inputDim[0]-kernelSize)/stride + 1;
        for(int depth = 0; depth < inputDim[2]; depth++){
                for(int x_start = 0; x_start < (inputDim[0]-kernelSize)/stride + 1 ; ++x_start){
                        for(int y_start = 0; y_start < (inputDim[1]-kernelSize)/stride + 1; ++y_start){
                                array<long, 3> inputStart = {x_start*stride, y_start*stride, depth};
                                array<long, 3> inputEnd = {kernelSize, kernelSize, 1};
                                array<long, 2> outStart = {depth*kernelSize*kernelSize, x_start*((inputDim[0]-kernelSize)/stride + 1) + y_start};
                                array<long, 2> outEnd = {kernelSize*kernelSize, 1};
                                array<long, 2> reshapeDim = {kernelSize*kernelSize, 1};
                                Tensor<float, 3> inputSliced = input.slice(inputStart, inputEnd);
                                Tensor<float, 2> flattened(kernelSize*kernelSize, 1);
                                for(int i = 0; i < kernelSize ; ++i){
                                        for(int j = 0; j < kernelSize; ++j){
                                                flattened(i*kernelSize + j, 0) = inputSliced(i, j, 0);
                                        }
                                }
                                outputMat.slice(outStart, outEnd) = flattened;
                        }
                }
        }


        for(int depth = 0; depth < inputDim[2]; ++depth){
                for(int i = 0; i < kernelSize; ++i){
                        for(int j = 0; j < kernelSize; ++j){
                                kernelRow(i*kernelSize + j + depth*kernelSize*kernelSize, 0) = kernel(i, j, depth, 0);
                        }
                }
        }
}


template <>
void Node<Tensor<float, 4>>::back(){
        switch(op){
                case opType::e_conv:{
                        //std::cout << "In conv backprop" << std::endl;
                        //std::cout << gradient << std::endl;
                        const Tensor<float, 4>::Dimensions& gradientDim = gradient.dimensions();
                        array<int, 3> dims = {0, 1, 2};
                        const Tensor<float, 4>::Dimensions& kernelDim = parents[1]->value.dimensions();
                        int kernelSize = kernelDim[0];
                        int numFilters = kernelDim[3];
                        const Tensor<float, 4>::Dimensions& inputDim = parents[0]->value.dimensions();
                        int height = inputDim[0];
                        int width = inputDim[1];
                        int depth = inputDim[2];
                        int numExamples = inputDim[3];


                        Tensor<float, 4> inputParentGradient(inputDim);
                        inputParentGradient.setZero();
                        Tensor<float, 4> filterParentGradient(kernelDim);
                        filterParentGradient.setZero();


                        for(int n = 0; n < numExamples; ++n){ 
                                
                                //std::cout << "In example " << n << std::endl;
                        
                                for(int f = 0; f < numFilters; ++f){
                                        
                                        //std::cout << "In filter " << f << std::endl;


                                        // Input slicing bounds
                                        array<long, 4> inputStart = {0, 0, 0, n};
                                        array<long, 4> inputExtent = {height, width, depth, 1};
                                        
                                        // Kernel slicing bounds        
                                        array<long, 4> kernelStart = {0, 0, 0, f};
                                        array<long, 4> kernelExtent = {kernelSize, kernelSize, depth, 1};


                                        // Gradient slicing bounds
                                        array<long, 4> gradientStart = {0, 0, f, n};
                                        array<long, 4> gradientExtent = {((height-kernelSize)/1 + 1), ((width-kernelSize)/1 + 1), 1, 1};


                                        // Sliced Kernel, Input and Gradient
                                        Tensor<float, 4> currKernel = parents[1]->value.slice(kernelStart, kernelExtent);
                                        Tensor<float, 4> currInput  = parents[0]->value.slice(inputStart, inputExtent);
                                        Tensor<float, 4> slicedGradient =  gradient.slice(gradientStart, gradientExtent);


                                        //std::cout << "Input image " << std::endl;
                                        //std::cout << currInput << std::endl;


                                        // im2col on input and kernel. Flatten Gradient
                                        Tensor<float, 2> inputMat(kernelSize*kernelSize*depth, ((height-kernelSize)/1 + 1) * ((width-kernelSize)/1 + 1)), inputFilter(kernelSize*kernelSize*depth, 1);
                                        im2col(currInput, currKernel, inputMat, inputFilter);
                                        Tensor<float, 2> flattenGradient(((height-kernelSize)/1 + 1)*((width-kernelSize)/1 + 1), 1);
                                        for(int col = 0; col < ((width-kernelSize)/1 + 1); ++col){
                                                for(int row = 0; row < ((height-kernelSize)/1 + 1); ++row){
                                                        flattenGradient(row*((height-kernelSize)/1 + 1) + col, 0) = slicedGradient(row, col, 0, 0);
                                                }
                                        }
/*
                                        std::cout << "Flattened input Gradient " << std::endl;
                                        std::cout << flattenGradient << std::endl;
                                        std::cout << "im2col image " << std::endl;
                                        std::cout << inputMat << std::endl;
                                        std::cout << "im2col filter " << std::endl;
                                        std::cout << inputFilter << std::endl;
*/


                                        // Compute gradient for input and kernel using backprop for multiply
                                        array<long, 2> shuffle = {1, 0};
                                        Tensor<float, 2> inputFilterTranspose = inputFilter.shuffle(shuffle);
                                        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
                                        Eigen::array<Eigen::IndexPair<int>, 1> transpose_product_dims = { Eigen::IndexPair<int>(0, 1) };
                                        Tensor<float, 2> filterGradient = inputMat.contract(flattenGradient, product_dims);        
                                        Tensor<float, 2> inputGradient = flattenGradient.contract(inputFilterTranspose, product_dims);
/*
                                        std::cout << "Flattened filter parent gradient " << std::endl;
                                        std::cout << filterGradient << std::endl;
                                        std::cout << "Flattened input parent gradient " << std::endl;
                                        std::cout << inputGradient << std::endl;
*/


                                        // run col2im on gradients and set gradients for each parent
                                        Tensor<float, 4> filterGradientIter(kernelDim[0], kernelDim[1], kernelDim[2], 1);
                                        Tensor<float, 4> inputGradientIter(inputDim[0], inputDim[1], inputDim[2], 1);
                                        inputGradientIter.setZero();
                                        filterGradientIter.setZero();
                                        col2im(inputGradientIter, filterGradientIter, inputGradient, filterGradient);
                                        inputParentGradient.slice(inputStart, inputExtent) += inputGradientIter;
                                        filterParentGradient.slice(kernelStart, kernelExtent) += filterGradientIter;


                                        //std::cout << inputFilterTranspose.contract(inputMat, product_dims)  << std::endl;
/*
                                        std::cout << "Input Gradient " << std::endl;
                                        std::cout << inputParentGradient << std::endl;


                                        std::cout << "Filter Gradient " << std::endl;
                                        std::cout << filterParentGradient << std::endl;
*/
                                }
                        }
                        gradientParent[parents[0]] = inputParentGradient;
                        gradientParent[parents[1]] = filterParentGradient;
                        //exit(0);
                        return;
                }
                case opType::e_maxpool:{
//                        std::cout << "In maxpool backprop" << std::endl;
                        std::shared_ptr<Node<MatrixXf>>* cMat = reinterpret_cast<std::shared_ptr<Node<MatrixXf>>*>(&children[0]);
                        gradient = cMat->get()->flattenGradient;
/*
                        std::cout << "gradient " << std::endl;
                        std::cout << gradient << std::endl;
*/
/*
                        std::cout << "Evaluated value " << std::endl;
                        std::cout << value << std::endl;
                        std::cout << "Parent0 " << std::endl;
                        std::cout << parents[0]->value << std::endl;        
                        std::cout << "Chosen indices " << std::endl;
                        std::cout <<  argMaxCache << std::endl;
*/                        
                        int poolSize = 2;
                        int poolStride = 1;
                        const Tensor<float, 4>::Dimensions& inputDim = parents[0]->value.dimensions();
                        Tensor<float, 4> finalGradient(inputDim);        
                        finalGradient = finalGradient.setZero();


                        //int numStrides0 = inputDim[0]/poolSize+1;
                        //int numStrides1 = inputDim[1]/poolSize+1;
                        int numStrides0 = inputDim[0]/poolSize;
                        int numStrides1 = inputDim[1]/poolSize;
                        
                        for(int n = 0; n < inputDim[3]; ++n){ 
                                for(int f = 0; f < inputDim[2]; ++f){
                                        //for(int p0 = 0; p0 < numStrides0; p0+=poolSize){
                                        for(int p0 = 0; p0 < numStrides0; p0+=poolStride){
                                                //for(int p1 = 0; p1 < numStrides1; p1+=poolSize){
                                                for(int p1 = 0; p1 < numStrides1; p1+=poolStride){
                                                        array<long, 4> gradientStart = {p0*poolSize, p1*poolSize, f, n};
                                                        array<long, 4> gradientExtent = {poolSize, poolSize, 1, 1};
                                                        Tensor<float, 4> gradientSlice = finalGradient.slice(gradientStart, gradientExtent);
/*
                                                        std::cout << "gradientSlice input " << std::endl;
                                                        std::cout << gradientSlice << std::endl;
*/
                                                        //gradientSlice(argMaxCache(p0/poolSize,p1/poolSize,f,n)) = gradient(p0/poolSize,p1/poolSize,f,n);
                                                        gradientSlice(argMaxCache(p0, p1, f, n)) += gradient(p0, p1, f, n);
                                                        //std::cout << "gradientSlice output " << std::endl;
                                                        //std::cout << gradientSlice << std::endl;
                                                        finalGradient.slice(gradientStart, gradientExtent) = gradientSlice;
                                                        }
                                                }
                                        }
                                }
                        gradientParent[parents[0]] = finalGradient;
                        //std::cout << finalGradient << std::endl;
                        //exit(0);
                        return;
                }
                case opType::e_relu:{        
                        Eigen::Tensor<float, 4> zeroConst(parents[0]->value.dimensions());
                        zeroConst.setConstant(0);
                        Tensor<bool, 4> parent0Cmp = parents[0]->value >= zeroConst;        
                        Tensor<float, 4> parent0CmpFloat = parent0Cmp.cast <float>();
                        Tensor<float, 4> finalGradientFloat = parent0CmpFloat*gradient;
                        //Eigen::Tensor<float, 4> finalGradientFloat = finalGradient.cast <float>();
/*
                        std::cout << "Inside relu backprop " << std::endl;
                        std::cout << parent0Cmp << std::endl;
                        std::cout << parent0CmpFloat << std::endl;
                        std::cout << finalGradientFloat << std::endl;
*/
                        gradientParent[parents[0]] = finalGradientFloat;
                        return;
                }
                case opType::e_nop:{
                        return;
                }
                default:{
                        std::cout << "Unsupported op for Tensor<float, 4>" << std::endl;
                        exit(0);
                        return;
                }
        }
}


template <>
void Node<MatrixXf>::back(){
        switch(op){
                case opType::e_flatten:{
                        //std::cout << "Input flatten gradient is " << gradient << std::endl;
                        Node<MatrixXf>* parentNodeMatrix = parents[0];
                        Node<Tensor<float, 4>>* parentNode = reinterpret_cast<Node<Tensor<float, 4>>*>(parentNodeMatrix);
                        
                        const Tensor<float, 4>::Dimensions& parentDim = parentNode->value.dimensions();
                        Tensor<float, 4> finalGradient(parentDim);
                        finalGradient.setConstant(0);
                        
                        for(int numExamples = 0; numExamples < parentDim[3]; numExamples++){
                                MatrixXf oneExample = gradient.row(numExamples);
                                Eigen::TensorMap<Tensor<float, 3>> tensorGradient(oneExample.data(), parentDim[0], parentDim[1], parentDim[2]);
                                finalGradient.chip(numExamples, 3) = tensorGradient;
                        }
                        flattenGradient = finalGradient;
                        MatrixXf* finalGradientMat = reinterpret_cast<MatrixXf*>(&finalGradient);
                        //std::cout << "Flatten Store " << *finalGradientMat << std::endl;
                        gradientParent[parents[0]] = *finalGradientMat;
                        return;
                }
                case opType::e_sum:{
                        MatrixXf localGradient =  MatrixXf::Ones(parents[0]->value.rows(),parents[0]->value.cols());
                        MatrixXf finalGradient = localGradient;
                        finalGradient *= gradient(0,0);
                        gradientParent[parents[0]] = finalGradient;
                        return;
                }
                case opType::e_emul:{        


                        gradientParent[parents[1]] = parents[0]->value.cwiseProduct(gradient);
                        gradientParent[parents[0]] = parents[1]->value.cwiseProduct(gradient);
                        return;
                }
                case opType::e_plus:{


                        int parent0Rows = parents[0]->value.rows();
                        int parent0Cols = parents[0]->value.cols();
                        int parent1Rows = parents[1]->value.rows();
                        int parent1Cols = parents[1]->value.cols();
                        int gradientRows = gradient.rows();
                        int gradientCols = gradient.cols();
                
                        //std::cout << "parent0 " << parent0Rows << " " << parent0Cols << std::endl;        
                        //std::cout << "parent1 " << parent1Rows << " " << parent1Cols << std::endl;        
                        //std::cout << "gradient " << gradientRows << " " << gradientCols << std::endl;        
                        if(parent0Rows == gradientRows && parent0Cols == gradientCols) {gradientParent[parents[0]] = gradient;}
                        else if(parent0Rows == gradientRows && parent0Cols == 1){gradientParent[parents[0]] = gradient.rowwise().sum();}
                        else if(parent0Cols == gradientCols && parent0Rows == 1){gradientParent[parents[0]] = gradient.colwise().sum();}
                        else { std::cout << "Invalid dimensions" << std::endl;}


                        if(parent1Rows == gradientRows && parent1Cols == gradientCols) {gradientParent[parents[1]] = gradient;}
                        else if(parent1Rows == gradientRows && parent1Cols == 1){gradientParent[parents[1]] = gradient.rowwise().sum();}
                        else if(parent1Cols == gradientCols && parent1Rows == 1){gradientParent[parents[1]] = gradient.colwise().sum();}
                        else { std::cout << "Invalid dimensions" << std::endl;}


                        //std::cout << "gradientParent0 for " << parents[0] << " is " << gradientParent[parents[0]] << std::endl;
                        //std::cout << "gradientParent1 for " << parents[1] << " is "  << gradientParent[parents[1]] << std::endl;
                        return;        
                }
                case opType::e_minus:{        
                        int parent0Rows = parents[0]->value.rows();
                        int parent0Cols = parents[0]->value.cols();
                        int gradientRows = gradient.rows();
                        int gradientCols = gradient.cols();
                        if(parent0Rows == gradientRows && parent0Cols == gradientCols){
                                gradientParent[parents[0]] = gradient;
}
                        else if(parent0Rows == gradientRows && parent0Cols == 1){
                                gradientParent[parents[0]] = gradient.rowwise().sum();
}
                        else if(parent0Cols == gradientCols && parent0Rows == 1){
                                gradientParent[parents[0]] = gradient.colwise().sum();
}
                                gradientParent[parents[1]] = -gradient;
                        return;
                }
                case opType::e_log:{        
                        gradientParent[parents[0]] = gradient.cwiseProduct(parents[0]->value.cwiseInverse());
                        return;
                }
                case opType::e_colsum:{        
                        gradientParent[parents[0]] = MatrixXf::Ones(parents[0]->value.rows(), parents[0]->value.cols());
                        VectorXf V(Map<VectorXf>(gradient.data(), gradient.rows()*gradient.cols()));
                        gradientParent[parents[0]] =  gradientParent[parents[0]].array().colwise() * V.array() ;
                        return;
                }
                case opType::e_exp:{        
                        gradientParent[parents[0]] = parents[0]->value.array().exp() * gradient.array();
                        
                        return;
                }
                case opType::e_mul:{        
/*
                        std::cout << "e_mul input gradient " << gradient << std::endl;
                        std::cout << "e_mul W input " << parents[1]->value << std::endl;
                        std::cout << "e_mul X input " << parents[0]->value  << std::endl;
*/
                        gradientParent[parents[0]] =  gradient * parents[1]->value.transpose();
                        gradientParent[parents[1]] =  parents[0]->value.transpose() * gradient;
/*
                        std::cout << "e_mul gradient parent0 " << gradientParent[parents[0]] << std::endl;
                        std::cout << "e_mul gradient parent1 " << gradientParent[parents[1]] << std::endl;
*/
                        return;
                }
                case opType::e_relu:{
                        //auto parent0Cmp = parents[0]->value.array() == value.array();
                        auto parent0Cmp = parents[0]->value.array() >= 0;
                        auto parent0Float = parent0Cmp.template cast<float>();
                        gradientParent[parents[0]] = parent0Float.matrix().cwiseProduct(gradient);
                        return;
                }
                case opType::e_max:{        
                        if(parents[0]->value.size() > 1){
                                auto parent0Cmp = parents[0]->value.array() == value.array();
                                auto parent0Float = parent0Cmp.template cast<float>();
                                gradientParent[parents[0]] = parent0Float.matrix().cwiseProduct(gradient);
                        } else {
                                auto parent0Cmp = value.array() == parents[0]->value(0,0);
                                auto parent0Float = parent0Cmp.template cast<float>();
                                gradientParent[parents[0]] = parent0Float.matrix().cwiseProduct(gradient);
                        }
                        if(parents[1]->value.size() > 1){
                                auto parent1Cmp = parents[1]->value.array() == value.array();
                                auto parent1Float = parent1Cmp.template cast<float>();
                                gradientParent[parents[1]] = parent1Float.matrix().cwiseProduct(gradient);
                        } else {
                                auto parent1Cmp = value.array() == parents[1]->value(0,0);
                                //std::cout << "parent1Cmp " <<  parent1Cmp << std::endl;
                                auto parent1Float = parent1Cmp.template cast<float>();
                                gradientParent[parents[1]] = parent1Float.matrix().cwiseProduct(gradient);
                        }
                        return;
                }
                default: { }
        }        
}


template <>
void Node<MatrixXf>::update(){
        if(op != opType::e_nop || !variable) {return;}
        value = value - 0.01*gradient;
        gradient = MatrixXf::Zero(0,0);
}


template <>
void Node<Tensor<float, 4>>::update(){
        if(op != opType::e_nop || !variable) {return;}
        value = value - 0.01*gradient;
        gradient.setZero();
}


template <typename T>
void Node<T>::update(){
}


Node<MatrixXf>& flatten(Node<Tensor<float, 4>>& first){
        std::vector<Node<Tensor<float, 4>>*> nodesVector = {&first};
        return storeFlatten(opType::e_flatten, nodesVector);
}


Node<MatrixXf>& storeFlatten(opType op, std::vector<Node<Tensor<float, 4>>*>& nodesVector){


        std::shared_ptr<Node<MatrixXf>> newNode = std::shared_ptr<Node<MatrixXf>>(new Node<MatrixXf>());        
        newNode->op = op;
        newNode->variable = false;


        // Store both parents to node as shared pointers
        for(Node<Tensor<float, 4>>* n: nodesVector){
                //std::cout << "Parents are: " << n << std::endl;
                newNode->parentType.push_back(1);
                newNode->parents.push_back(reinterpret_cast<Node<MatrixXf>*>(n));
                std::shared_ptr<Node<Tensor<float, 4>>>* castNode = reinterpret_cast<std::shared_ptr<Node<Tensor<float, 4>>>*>(&newNode);
                n->children.push_back(*castNode);
                n->childType.push_back(0);
        }
        //std::cout << "Here" << std::endl;        
        //std::cout << "Newnode parents: " << newNode->parents.size() << std::endl;
        return *newNode;
}
template class Node<NodeBase>;
template class Node<MatrixXf>;
template class Node<Tensor<float, 2>>;
template class Node<Tensor<float, 3>>;
template class Node<Tensor<float, 4>>;