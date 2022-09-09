#include "src/NeuralNetwork/NeuralNetwork.hpp"

#include <typeinfo>
#include <chrono>
#include <thread>

using utility::MatrixIt;
using utility::Timer;

using namespace linear_algebra;
using namespace neural_network::activation;
using namespace neural_network::loss;
using namespace neural_network::layer;

class NeuralNet
{
    typedef linear_algebra::Matrix<double, 4, 1> OutputTp;
    typedef linear_algebra::Matrix<double, 8, 1> InputTp;

public:
    OutputTp forward(const InputTp &x)
    {
        auto l1_out = l1_.forward(x);
        auto l2_out = l2_.forward(l1_out);
        auto l3_out = l3_.forward(l2_out);
        this->last_pred_ = l4_.forward(l3_out);
        return this->last_pred_;
    }

    void backward(const OutputTp& target)
    {
        Timer t1;
        auto grad = MSE<4>::grad(this->last_pred_, target);
    }

private:
    Linear<8, 16, ReLU<>> l1_;
    Linear<16, 16> l2_;
    Linear<16, 16> l3_;
    Linear<16, 4, ReLU<>> l4_;
    OutputTp last_pred_;
};

int main()
{
    try
    {
        {
            NeuralNet nn1;
            Timer t1;

            std::cout << "NN Size = " << sizeof(nn1) << '\n';
            std::cout << nn1.forward(Matrix<double, 8, 1>().fill_random(1, 10)) << '\n';
            nn1.backward(Matrix<double, 4, 1>(0));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
