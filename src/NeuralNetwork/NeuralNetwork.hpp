#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "../LinearAlgebra/LinearAlgebra.hpp"

namespace neural_network
{
    typedef int_fast64_t SizeType;

    namespace activation
    {
        class ReLU
        {
        };

        class Tanh
        {
        };

        class Default
        {
        };
    }

    namespace layer
    {
        template <typename ActiFunc, SizeType n_inputs, SizeType n_outputs>
        class Linear
        {
        public:
            Linear() = default;
            ~Linear() = default;
            Linear(const Linear &r_lnnet) = default;
            Linear(Linear &&r_lnnet) = default;
            Linear &operator=(const Linear &r_lnnet) = default;
            Linear &operator=(Linear &&r_lnnet) = default;

        protected:
        private:
            linear_algebra::Matrix<double, n_inputs, n_outputs> data_;
            ActiFunc acti_func_;
        };
    }

}

#endif /* NEURAL_NETWORK_HPP */