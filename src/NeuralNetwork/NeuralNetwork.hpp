#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Computation/Computation.hpp"
#include "Exception/Exception.hpp"
#include "Function/Function.hpp"
#include "Layer/Layer.hpp"

#include "../LinearAlgebra/LinearAlgebra.hpp"
#include "../Utility/Utility.hpp"

#include <type_traits>

namespace neural_network
{
    typedef int_fast64_t SizeTp;

    namespace activation
    {
        struct ActivationBase
        {
        };

        struct Identity : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                return std::move(output);
            }
        };

        template <typename NextFunc = Identity>
        struct ReLU : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                relu(std::begin(output), std::end(output));
                return std::move(NextFunc()(std::move(output)));
            }
        };

        template <typename NextFunc = Identity>
        struct Tanh : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                tanh(std::begin(output), std::end(output));
                return std::move(NextFunc()(std::move(output)));
            }
        };

        template <typename NextFunc = Identity>
        struct Sigmoid : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                sigmoid(std::begin(output), std::end(output));
                return std::move(NextFunc()(std::move(output)));
            }
        };

        template <typename NextFunc = Identity>
        struct Softmax : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                auto it_begin = std::begin(output);
                auto it_end = std::end(output);
                while (it_begin != it_end)
                    *(it_begin++) = std::max(0.0, *(it_begin));
                return std::move(NextFunc()(std::move(output)));
            }
        };
    }

    namespace layer
    {
        class LayerBase
        {
        public:
            LayerBase() = default;
            ~LayerBase() = default;
            LayerBase(const LayerBase &r_layer) = default;
            LayerBase(LayerBase &&r_layer) = default;
            LayerBase &operator=(const LayerBase &r_layer) = default;
            LayerBase &operator=(LayerBase &&r_layer) = default;

        protected:
            template <typename... T>
            void fill_random(T &...params)
            {
                (params.fill_random(-1.0, 1.0), ...);
            }

        private:
        };

        template <SizeTp n_inputs, SizeTp n_outputs, typename Func = activation::Identity>
        class Linear : public LayerBase
        {
            typedef linear_algebra::Matrix<double, 1, n_outputs> OutputTp;
            typedef linear_algebra::Matrix<double, 1, n_inputs> InputTp;
            typedef linear_algebra::Matrix<double, n_inputs, n_outputs> WeightTp;

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

            OutputTp forward(const InputTp &input) const
            {
                return func_(input * this->weight_ + this->bias_);
            }

            InputTp backward(const OutputTp &output_grad) const
            {
                return InputTp();
            }

        protected:
            WeightTp weight_;
            OutputTp bias_;
            Func func_;
        };
    }

}

#endif /* NEURAL_NETWORK_HPP */