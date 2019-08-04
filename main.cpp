#include <iostream>
#include <fstream>
#include "Graph.h"


// Input
#define WIDTH 16
#define HEIGHT 16
#define DEPTH 3
#define EXAMPLES 10
#define CLASSES 2


// Filter Dimensions
#define KERNELSIZE 2
#define NUMFILTERS 3
#define STRIDE 1


// Maxpool Dimensions
#define MAXPOOL_STRIDE 1


#define CEILING(x,y) (((x) + (y) - 1) / (y))


// Flatten Features
#define FEATSIZE CEILING((WIDTH-KERNELSIZE+1),2)*CEILING((HEIGHT-KERNELSIZE+1),2)*NUMFILTERS


int main(){


        Graph g;


        // Input Features


        Tensor<float, 4> XTensor(WIDTH, HEIGHT, DEPTH, EXAMPLES);
        XTensor.setRandom();
        Tensor<float, 4> constTensor(WIDTH, HEIGHT, DEPTH, EXAMPLES);
        constTensor.setConstant(0.5f);
        XTensor = XTensor - constTensor;
        Node<Tensor<float, 4>> XTNode(XTensor, false);


        // Labels


        MatrixXf YMat = MatrixXf::Zero(EXAMPLES, CLASSES);
        for(int i = 0; i < EXAMPLES; ++i){YMat(i, i%CLASSES) = 1;}
        Node<MatrixXf> Y(YMat, false);


        // Parameters


        // Conv Filter
        Tensor<float, 4> FTensor(KERNELSIZE, KERNELSIZE, DEPTH, NUMFILTERS);
        FTensor.setConstant(1.0f);
        //FTensor.setRandom();
        Node<Tensor<float, 4>> FTNode(FTensor);


        // Affine
        Node<MatrixXf> W(MatrixXf::Zero(FEATSIZE, CLASSES));
        Node<MatrixXf> X(MatrixXf::Random(EXAMPLES, FEATSIZE));
        Node<MatrixXf> B(MatrixXf::Zero(1, CLASSES));


        // Conv - Relu - Maxpool - Flatten


        Node<Tensor<float, 4>>& Co = conv(XTNode, FTNode);
        //Node<Tensor<float, 4>>& CoRel = relu(Co);
        Node<Tensor<float, 4>>& CoMax = maxPool2D(Co);
        Node<MatrixXf>& CoFlat = flatten(CoMax);
        Node<MatrixXf>& A = CoFlat * W;                 


        // Softmax
        Node<MatrixXf>& Z0 = A + B;
        Node<MatrixXf>& Z = relu(Z0);
        Node<MatrixXf>& softmax = nlog(sum(exp(Z), 1)) - Z;
        Node<MatrixXf>& loss = sum(emul(Y, softmax), -1);




        for(int iter = 0; iter < 100; iter++){
                g.eval(loss);


                std::cout << "loss " << loss.value << std::endl;


                g.backProp(loss);


/*
                std::cout << "FTNode " << FTNode.gradient << std::endl;
                std::cout << "Co " << Co.gradient << std::endl;
                std::cout << "CoMax " << CoMax.gradient << std::endl;
*/
                g.update(loss);
        }


}