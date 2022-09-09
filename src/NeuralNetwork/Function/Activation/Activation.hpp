#ifndef NEURAL_NETWORK_FUNCTION_ACTIVATION_HPP
#define NEURAL_NETWORK_FUNCTION_ACTIVATION_HPP

#include "../../../LinearAlgebra/LinearAlgebra.hpp"

#include <math.h>

namespace neural_network
{
    namespace activation
    {
        struct ActivationBase
        {
        };

        struct Identity : public ActivationBase
        {
            template <typename Tp>
            static Tp forward(Tp x) { return x; }

            template <typename Tp>
            static Tp backward(Tp x) { return x; }
        };

        template <typename NextFunc = Identity>
        struct ReLU : public ActivationBase
        {
            // No copy will be made if the argument is an rvalue.
            template <typename Tp>
            static Tp forward(Tp x)
            {
                relu(std::begin(x), std::end(x));

                // Since <x> is still an lvalue, std::move is needed.
                // The result given by the next function will be
                // automatically moved by the compiler. Hence, std::move
                // is not necessary.
                return NextFunc::forward(std::move(x));
            }

            template <typename Tp>
            static Tp backward(Tp x)
            {
                auto g = NextFunc::backward(x);
                auto g_it = std::begin(g);
                auto x_it = std::begin(x);

                // while(res_it != res_it_end)
                // {
                //     // if (*(li_it) == 0)
                //     //     throw std::runtime_error("Undefined gradient.");
                //     *(li_it)>=0?
                // }
                return g;
            }
        };

        template <typename NextFunc = Identity>
        struct Tanh : public ActivationBase
        {
            template <typename Tp>
            static Tp forward(const Tp &x)
            {
                tanh(std::begin(x), std::end(x));
                return NextFunc::forward(x);
            }

            template <typename Tp>
            static Tp backward(const Tp &output)
            {
                return output;
            }
        };

        template <typename NextFunc = Identity>
        struct Sigmoid : public ActivationBase
        {
            template <typename Tp>
            static Tp forward(const Tp &x)
            {
                sigmoid(std::begin(x), std::end(x));
                return NextFunc::forward(x);
            }

            template <typename Tp>
            static Tp backward(const Tp &output)
            {
                return output;
            }
        };
    }
}

#endif /* NEURAL_NETWORK_FUNCTION_ACTIVATION_HPP */