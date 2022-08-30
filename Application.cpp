// #include "src/LinearAlgebra/LinearAlgebra.hpp"
#include "src/NeuralNetwork/NeuralNetwork.hpp"

#include <typeinfo>
#include <chrono>
#include <thread>

using utility::MatrixIt;
using utility::Timer;

using namespace linear_algebra;
using namespace neural_network;
using namespace neural_network::activation;
using namespace neural_network::layer;

class NeuralNet
{
public:
    Matrix<double, 1, 10> forward(Matrix<double, 1, 16> input)
    {
        auto l1_out = l1_.forward(input);
        auto l2_out = l2_.forward(l1_out);
        auto l3_out = l3_.forward(l2_out);
        return l4_.forward(l3_out);
    }

    void backward()
    {
    }

protected:
    Linear<16, 256, Sigmoid<>> l1_;
    Linear<256, 256, ReLU<>> l2_;
    Linear<256, 256, Sigmoid<>> l3_;
    Linear<256, 10> l4_;
};

int main()
{
    try
    {
        NeuralNet nn1;
        Timer t;

        while (1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            Timer t1;
            std::cout << nn1.forward(Matrix<double, 1, 16>().fill_random(1, 10)) << '\n';
            std::cout << sizeof(nn1) << '\n';
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
