#ifndef NEURAL_NETWORK_LAYER_LINEAR_HPP
#define NEURAL_NETWORK_LAYER_LINEAR_HPP

#include "LayerBase.hpp"
#include "../Function/Function.hpp"

#include <iostream>

namespace neural_network
{
    namespace layer
    {
        template <SizeTp n_inputs, SizeTp n_outputs, typename Func = activation::Identity>
        class Linear : public LayerBase
        {
            typedef linear_algebra::Matrix<double, n_outputs, 1> OutputTp;
            typedef linear_algebra::Matrix<double, n_inputs, 1> InputTp;
            typedef linear_algebra::Matrix<double, n_outputs, n_inputs> WeightTp;

        public:
            Linear() { this->fill_random(this->weight_, this->bias_); }

            friend std::ostream &operator<<(std::ostream &os, const Linear &lin_layer)
            {
                os << "Weight = \n"
                   << lin_layer.weight_ << '\n'
                   << "Bias = \n"
                   << lin_layer.bias_ << '\n';
                return os;
            }

            OutputTp forward(const InputTp &x)
            {
                this->last_inp_ = x;

                // The result of the operations will be passed as an rvalue 
                // to the forward function by the compiler.
                return Func::forward(this->weight_ * x + this->bias_);
            }

            // InputTp backward(const OutputTp &grad, double learning_rate)
            // {
            //     auto w_grad = last_inp_.transpose() * grad;
            //     this->weight_ = this->weight_ - w_grad * learning_rate;
            //     this->bias_ = this->bias_ - grad * learning_rate;
            //     return this->weight_ * grad.transpose();
            // }

        protected:
            WeightTp weight_;
            OutputTp bias_;
            InputTp last_inp_;
        };
    }
}

#endif /* NEURAL_NETWORK_LAYER_LINEAR_HPP */