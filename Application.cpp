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
    typedef linear_algebra::Matrix<double, 1, 1> OutputTp;
    typedef linear_algebra::Matrix<double, 1, 1> InputTp;

public:
    OutputTp forward(const InputTp &x)
    {
        auto l1_out = l1_.forward(x);
        auto l2_out = l2_.forward(l1_out);
        auto l3_out = l3_.forward(l2_out);
        this->last_pred_ = l4_.forward(l3_out);
        return this->last_pred_;
    }

    void backward(const OutputTp &target)
    {
        double alpha = 0.1;
        auto grad = MSE<1>::grad(target, this->last_pred_);
        auto loss = MSE<1>::loss(target, this->last_pred_);
        this->loss = this->loss_value(std::begin(loss), std::end(loss));
        auto l4_grad = this->l4_.backward(grad, alpha);
        auto l3_grad = this->l3_.backward(l4_grad, alpha);
        auto l2_grad = this->l2_.backward(l3_grad, alpha);
        this->l1_.backward(l2_grad, alpha);
    }

    double loss;

private:
    template <typename ItTp>
    double loss_value(ItTp it_begin, ItTp it_end)
    {
        double res = 0.0;
        while (it_begin != it_end)
            res += pow(*(it_begin++), 2);
        return sqrt(res);
    }

    Linear<1, 16, ReLU<>> l1_;
    Linear<16, 16, ReLU<>> l2_;
    Linear<16, 16, ReLU<>> l3_;
    Linear<16, 1> l4_;
    OutputTp last_pred_;
};

int main()
{
    try
    {
        {
            NeuralNet nn1;
            std::cout << "NN Size = " << sizeof(nn1) << '\n';

            Timer t1;
            double ave_loss = 0.0;
            for (int i = 0;; i++)
            {
                auto input = Matrix<double, 1, 1>().fill_random(1, 100);
                std::cout << "Input:\n"
                          << input << '\n';
                std::cout << "Predicted:\n"
                          << nn1.forward(input) << '\n';
                nn1.backward(input);
                ave_loss += nn1.loss;
                if (i % 100000 == 0)
                {
                    std::cout << "Average Loss = " << ave_loss / 10000 << '\n';
                    ave_loss = 0.0;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
