#include "Graph.h"

// Forward Pass

void Graph::eval(Node<MatrixXf>& node){
    std::set<Node<NodeBase>*> explored;
    //std::set<Node<MatrixXf>*> explored;
    return _eval(node, explored);
}

void Graph::eval(Node<Tensor<float, 2>>& node){
    std::set<Node<NodeBase>*> explored;
    return _eval(node, explored);
}

void Graph::eval(Node<Tensor<float, 4>>& node){
    std::set<Node<NodeBase>*> explored;
    return _eval(node, explored);
}

void Graph::_eval(Node<MatrixXf>& node, std::set<Node<NodeBase>*>& explored){
    //std::cout << "Graph eval for MatrixXf node " << &node << std::endl;
    if(node.parents.size() == 0){ node.eval(); return;}
    std::vector <Node<MatrixXf>*> parents = node.parents;
    for(Node<MatrixXf>* n: parents){
        if(node.op == opType::e_flatten){
            Node<Tensor<float, 4>>* nTensor = reinterpret_cast<Node<Tensor<float, 4>>*>(n);
            _eval(*nTensor, explored); 
        } else {_eval(*n, explored);}
    }
    //Node<NodeBase>* baseNode = base(&node);
    Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
    if(explored.find(baseNode) == explored.end()){
        std::cout << "Eval op " << node.op << " for node "<< &node << std::endl;
        //MatrixXf& mat = node.eval();
        node.eval();
        //Node<NodeBase>* baseNode = base(&node);
        Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
        explored.insert(baseNode);
        //std::cout << mat << std::endl;
        return;
    }
    else{return;}
}

void Graph::_eval(Node<Tensor<float, 2>>& node, std::set<Node<NodeBase>*>& explored){
    //std::cout << "Graph eval for node " << &node << std::endl;
    if(node.parents.size() == 0){ node.eval(); return;}
    auto parents = node.parents;
    for(auto n: parents){ _eval(*n, explored);}
    //Node<NodeBase>* baseNode = base(&node);
    Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
    if(explored.find(baseNode) == explored.end()){
        std::cout << "Eval op " << node.op << " for node "<< &node << std::endl;
        //Tensor<float, 2>& mat = node.eval();
        node.eval();
        //Node<NodeBase>* baseNode = base(&node);
        Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
        explored.insert(baseNode);
        //std::cout << mat << std::endl;
        return;
    }
    else{return;}
}

void Graph::_eval(Node<Tensor<float, 4>>& node, std::set<Node<NodeBase>*>& explored){
    if(node.parents.size() == 0){ node.eval(); return;}
    std::vector<Node<Tensor<float, 4>>*> parents = node.parents;
    for(auto n: parents){ _eval(*n, explored);}
    Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
    if(explored.find(baseNode) == explored.end()){
        std::cout << "Eval op " << node.op << " for node "<< &node << std::endl;
         node.eval();
        //Node<NodeBase>* baseNode = base(&node);
        Node<NodeBase>* baseNode = reinterpret_cast<Node<NodeBase>*>(&node);
        explored.insert(baseNode);
        //std::cout << mat << std::endl;
        return;
    }
    else{return;}
}

// Backward Pass
// Only computes gradients. Does not update

void Graph::backProp(Node<MatrixXf>& node) {

    std::set<Node<MatrixXf>*> explored;
    node.gradient = MatrixXf::Ones(1,1);
    //std::cout << "First gradient is " <<  node.gradient << std::endl;
    std::queue<Node<MatrixXf>*> nodeQueue;
    std::queue<std::pair<Node<MatrixXf>*, int>> nodeQueuePair;
    nodeQueue.push(&node);
    nodeQueuePair.push(std::make_pair(&node, 0));

    while(!nodeQueue.empty()){

        Node<MatrixXf>* node = nodeQueue.front();
        nodeQueue.pop();

        std::pair<Node<MatrixXf>*, int> nodePair = nodeQueuePair.front();
        nodeQueuePair.pop();
        
        int reinterpret = nodePair.second;
    
        if(!reinterpret){
            while( !node->areChildrenDone() ){
                if(nodeQueue.empty()){
                    std::cout << "Queue is empty" << std::endl;
                    exit(0);
                }
                nodeQueue.push(node);
                node = nodeQueue.front();
                //std::cout << "Popped " << node << std::endl;
                nodeQueue.pop();
            }
        }
        else{
            Node<Tensor<float, 4>>* nodeTensor = reinterpret_cast<Node<Tensor<float, 4>>*>(node);
            //std::cout << "Op " << nodeTensor->op << std::endl;
            while( !nodeTensor->areChildrenDone() ){
                if(nodeQueue.empty()){
                    std::cout << "Queue is empty" << std::endl;
                    exit(0);
                }
                nodeQueue.push(node);
                node = nodeQueue.front();
                std::cout << "Popped " << node << std::endl;
                nodeQueue.pop();
            }
        
        }

        if(reinterpret){
            //std::cout << "Reinterpreted" << std::endl;
            Node<Tensor<float, 4>>* nodeTensor = reinterpret_cast<Node<Tensor<float, 4>>*>(node);
            std::cout << "Back op with reinterpret " << nodeTensor->op << " for node " << nodeTensor << std::endl;
            clock_t startTime = clock();
            nodeTensor->back();
            std::cout << "Backward Pass for op " << nodeTensor->op << " " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << std::endl;
            
            std::vector<Node<Tensor<float, 4>>*> parents = nodeTensor->parents;
            int parentId = 0;
            for(auto nTensor: parents){
                //std::cout << "Computed gradient for parent " << nTensor << " being pushed is " <<  nodeTensor->gradientParent[nTensor] << std::endl;
                //std::cout << "Child Op : " << nTensor->op << std::endl;
                //std::cout << "Parent Type for node " << nTensor << " parentId " << parentId << " is " << nodeTensor->parentType[parentId] << std::endl;
                Node<MatrixXf>* n = reinterpret_cast<Node<MatrixXf>*>(nTensor);
                if(explored.find(n) == explored.end()){ 
                    nodeQueue.push(n);
                    nodeQueuePair.push(std::make_pair(n, nodeTensor->parentType[parentId]));
                    explored.insert(n); 
                }
                parentId++;
            }
        }
        
        else{
            std::cout << "Back op " << node->op << " for node " << node << std::endl;
            //std::cout << "Local gradient for this op " << std::endl;
            //std::cout << node->gradient << std::endl;
            clock_t startTime = clock();
            node->back();
            std::cout << "Backward Pass for op " << node->op << " " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << std::endl;
            std::vector<Node<MatrixXf>*> parents = node->parents;
            //if(node->op == opType::e_flatten){reinterpret = true;}
            int parentId = 0;
            for(auto n: parents){
                //std::cout << "Computed gradient for parent " << n << " being pushed is " <<  node->gradientParent[n] << std::endl;
                //std::cout << "Parent Type for node " << n << " parentId " << parentId << " is " << node->parentType[parentId] << std::endl;
                if(explored.find(n) == explored.end()){ 
                    //if(reinterpret) {std::cout << "Reinterpret pointer  " <<  n << std::endl;}
                    nodeQueue.push(n);
                    nodeQueuePair.push(std::make_pair(n, node->parentType[parentId]));
                    explored.insert(n); 
                }
                parentId++;
            }
        }
    }
    return;
}

void Graph::update(Node<MatrixXf>& node){
    
    std::set<Node<MatrixXf>*> explored;
    std::queue<std::pair<Node<MatrixXf>*, int>> nodeQueue;
    nodeQueue.push(std::make_pair(&node, 0));

    while(!nodeQueue.empty()){

        std::pair<Node<MatrixXf>*, int> nodePair = nodeQueue.front();
        nodeQueue.pop();
        Node<MatrixXf>* node = nodePair.first;

        if(nodePair.second){

            Node<Tensor<float, 4>>* nodeTensor = reinterpret_cast<Node<Tensor<float, 4>>*>(node);
            nodeTensor->update();
            
            std::vector<Node<Tensor<float, 4>>*> parents = nodeTensor->parents;

            int parentId = 0;
            for(auto nTensor: parents){
                Node<MatrixXf>* n = reinterpret_cast<Node<MatrixXf>*>(nTensor);
                if(explored.find(n) == explored.end()){ 
                    nodeQueue.push(std::make_pair(n, nodeTensor->parentType[parentId]));
                    explored.insert(n); 
                }
                parentId++;
            }
        }
        else{
            //std::cout << "Update on " << node << std::endl;
            node->update();

            std::vector<Node<MatrixXf>*> parents = node->parents;

            int parentId = 0;
            for(auto n: parents){
                if(explored.find(n) == explored.end()){ 
                    nodeQueue.push(std::make_pair(n, node->parentType[parentId]));
                    explored.insert(n); 
                }
            }
        }
    }
    return;
}



