#include <set>
#include <queue>
#include "Node.h"


class Graph {
        public:
                Graph(){};
                void eval(Node<MatrixXf>& node);
                void eval(Node<Tensor<float, 2>>& node);
                void eval(Node<Tensor<float, 4>>& node);


                void backProp(Node<MatrixXf>& node);
                void update(Node<MatrixXf>& node);
        private:
                void  _eval(Node<MatrixXf>& node, std::set<Node<NodeBase>*>& explored);
                void _eval(Node<Tensor<float, 2>>& node, std::set<Node<NodeBase>*>& explored);
                void _eval(Node<Tensor<float, 4>>& node, std::set<Node<NodeBase>*>& explored);
};