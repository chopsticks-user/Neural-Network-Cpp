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
            static Tp backward(Tp x, const Tp &last_inp) { return x; }
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

            template <typename GradTp>
            static GradTp backward(GradTp grad, const GradTp &last_inp)
            {
                grad = NextFunc::backward(std::move(grad), last_inp);
                auto li_it = std::begin(last_inp);
                auto g_it = std::begin(grad);
                auto g_it_end = std::end(grad);

                while (g_it != g_it_end)
                {
                    if (*g_it > 0)
                        *g_it = *li_it;
                    else
                        *g_it = 0;
                    g_it++;
                    li_it++;
                }
                return grad;
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