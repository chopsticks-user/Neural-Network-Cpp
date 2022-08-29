// #include "src/LinearAlgebra/LinearAlgebra.hpp"
#include "src/NeuralNetwork/NeuralNetwork.hpp"

#include <typeinfo>

using utility::MatrixIt;
using utility::Timer;

using namespace linear_algebra;
using namespace neural_network;
using namespace neural_network::activation;
using namespace neural_network::layer;

class up
{
private:
    std::unique_ptr<int> ptr_;

public:
    up() : ptr_(std::make_unique<int>()){};
    up(const up &other) : ptr_(std::make_unique<int>(*(other.ptr_.get()))){};
};

int main()
{

    try
    {
        Timer t;
        Matrix<double, 1, 10> inp;
        Linear<Default, 10, 16> l1;
        Linear<Default, 16, 16> l2;
        Linear<Default, 16, 16> l3;
        Linear<Default, 16, 2> l4;

        inp.fill_random(1, 10);
        auto l1_out = l1.forward(inp);
        auto l2_out = l2.forward(l1_out);
        auto l3_out = l3.forward(l2_out);
        auto l4_out = l4.forward(l3_out);

        std::cout << l4_out << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
