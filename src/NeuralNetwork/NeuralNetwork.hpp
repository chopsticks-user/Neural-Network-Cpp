#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

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

        struct Default : public ActivationBase
        {
            template <typename OutputTp>
            OutputTp &&operator()(OutputTp &&output) const
            {
                try
                {
                    return std::move(output);
                }
                catch (const std::exception &e)
                {
                    std::cerr << e.what() << '\n';
                    return std::move(output);
                }
            }
        };

        template <typename NextFunc = Default>
        struct ReLU : public ActivationBase
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

        template <typename NextFunc = Default>
        struct Tanh : public ActivationBase
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

        template <typename NextFunc = Default>
        struct Sigmoid : public ActivationBase
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

        template <typename NextFunc = Default>
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
        };

        template <SizeTp n_inputs, SizeTp n_outputs, typename Func = activation::Default>
        class Linear : public LayerBase
        {
        public:
            Linear()
            {
                static_assert(std::is_base_of_v<activation::ActivationBase, Func>);
                this->fill_random();
            }

            ~Linear() = default;

            Linear(const Linear &r_lin_layer)
                : weight_(r_lin_layer.weight_), bias_(r_lin_layer.bias_){};

            Linear(Linear &&r_lin_layer) = default;

            Linear &operator=(Linear &r_lin_layer)
            {
                this->weight_ = r_lin_layer.weight_;
                this->bias_ = r_lin_layer.bias_;
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
            forward(const linear_algebra::Matrix<double, 1, n_inputs> &input) const
            {
                return func_(input * this->weight_ + this->bias_);
            }

        protected:
            void fill_random()
            {
                this->weight_.fill_random(-1.0, 1.0);
                this->bias_.fill_random(-1.0, 1.0);
            }

            linear_algebra::Matrix<double, n_inputs, n_outputs> weight_;
            linear_algebra::Matrix<double, 1, n_outputs> bias_;
            Func func_;
        };
    }

}

#endif /* NEURAL_NETWORK_HPP */