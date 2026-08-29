#include <iostream>

#include "edgert/tensor.h"

int main() {
    edgert::Tensor t({2, 3}, edgert::DataType::Float32);
    t.fill(1.0F);
    t.at({0, 0}) = 42.0F;

    std::cout << "EdgeRT demo\n";
    std::cout << "shape: [";
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.shape()[i];
        if (i + 1 < t.shape().size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "numel: " << t.numel() << "\n";
    std::cout << "nbytes: " << t.nbytes() << "\n";
    std::cout << "t[0,0] = " << t.at({0, 0}) << "\n";
    std::cout << "t[1,2] = " << t.at({1, 2}) << "\n";
    return 0;
}