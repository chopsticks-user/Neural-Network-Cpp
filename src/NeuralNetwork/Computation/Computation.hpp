#ifndef NEURAL_NETWORK_COMPUTATION_HPP
#define NEURAL_NETWORK_COMPUTATION_HPP

#include <cmath>

namespace neural_network
{
    double sigmoid(double val)
    {
        return 1.0 / (1 + exp(-val));
    }

    template <typename ItTp>
    void sigmoid(ItTp it_begin, ItTp it_end)
    {
        while (it_begin != it_end)
            *(it_begin++) = sigmoid(*(it_begin));
    }

    double relu(double val)
    {
        return std::max(0.0, val);
    }

    template <typename ItTp>
    void relu(ItTp it_begin, ItTp it_end)
    {
        while (it_begin != it_end)
            *(it_begin++) = relu(*(it_begin));
    }

    double tanh(double val)
    {
        return std::tanh(val);
    }

    template <typename ItTp>
    void tanh(ItTp it_begin, ItTp it_end)
    {
        while (it_begin != it_end)
            *(it_begin++) = tanh(*(it_begin));
    }
}

#endif /* NEURAL_NETWORK_COMPUTATION_HPP */