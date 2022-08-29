// #include "src/LinearAlgebra/LinearAlgebra.hpp"
#include "src/NeuralNetwork/NeuralNetwork.hpp"

#include <typeinfo>

using utility::MatrixIt;
using utility::Timer;

using namespace linear_algebra;
using namespace neural_network;
using namespace neural_network::activation;
using namespace neural_network::layer;

template <typename U = float, typename T = int>
class A
{
    U aa;
};

int main()
{

    try
    {
        A a;
        Timer t;
        Matrix<double, 1, 64> inp;
        Linear<64, 128, Sigmoid<>> l1;
        Linear<128, 128, ReLU<>> l2;
        Linear<128, 128, Tanh<>> l3;
        Linear<128, 16, Softmax<>> l4;

        inp.fill_random(1, 10);
        auto l1_out = l1.forward(inp);
        auto l2_out = l2.forward(l1_out);
        auto l3_out = l3.forward(l2_out);
        auto l4_out = l4.forward(l3_out);

        std::cout << l4_out << '\n'
                  << sizeof(l4) << '\n'
                  << sizeof(l4_out) << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
