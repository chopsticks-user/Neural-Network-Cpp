// #include "src/LinearAlgebra/LinearAlgebra.hpp"
#include "src/NeuralNetwork/NeuralNetwork.hpp"

#include <typeinfo>

using utility::MatrixIt;
using utility::Timer;

using namespace linear_algebra;
using namespace neural_network;
using namespace neural_network::activation;
using namespace neural_network::layer;

int main()
{

    try
    {
        Timer t;
        Linear<Default, 16, 16> input_layer;
        std::cout << sizeof(input_layer) << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
