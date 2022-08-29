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
            Linear() { this->fill_random(); }

            ~Linear() = default;

            Linear(const Linear &r_lin_layer)
                : weight_(r_lin_layer.weight_), bias_(r_lin_layer.bias_), activ_f_(r_lin_layer.activ_f_){};

            Linear(Linear &&r_lin_layer) = default;

            Linear &operator=(Linear &r_lin_layer)
            {
                this->weight_ = r_lin_layer.weight_;
                this->bias_ = r_lin_layer.bias_;
                this->activ_f_ = r_lin_layer.activ_f_;
            };

            Linear &operator=(Linear &&r_lin_layer) = default;

            friend std::ostream &operator<<(std::ostream &os, const Linear &lin_layer)
            {
                os << "Weight = \n"
                   << lin_layer.weight_ << '\n';
                os << "Bias = \n"
                   << lin_layer.bias_ << '\n';
                return os;
            }

            linear_algebra::Matrix<double, 1, n_outputs>
            forward(const linear_algebra::Matrix<double, 1, n_inputs>& input) const
            {
                return input * this->weight_ + this->bias_;
            }

        protected:
            void fill_random()
            {
                this->weight_.fill_random(0.0, 1.0);
                this->bias_.fill_random(0.0, 1.0);
            }

            linear_algebra::Matrix<double, n_inputs, n_outputs> weight_;
            linear_algebra::Matrix<double, 1, n_outputs> bias_;
            ActiFunc activ_f_;
        };
    }

}

#endif /* NEURAL_NETWORK_HPP */