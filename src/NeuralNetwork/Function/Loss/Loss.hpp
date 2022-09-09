#ifndef NEURAL_NETWORK_FUNCTION_LOSS_HPP
#define NEURAL_NETWORK_FUNCTION_LOSS_HPP

#include "../../../LinearAlgebra/LinearAlgebra.hpp"

#include <math.h>

namespace neural_network
{
    namespace loss
    {
        template <unsigned int GradSize>
        struct LossBase
        {
            using GradTp = linear_algebra::Matrix<double, GradSize, 1>;

            LossBase() = default;
        };

        template <unsigned int GradSize>
        struct MSE : public LossBase<GradSize>
        {
            typedef typename LossBase<GradSize>::GradTp GradTp;

            static GradTp loss(const GradTp &target, const GradTp &pred)
            {
                GradTp l;
                const auto *t_it = std::begin(target);
                const auto *p_it = std::begin(pred);
                auto l_it = std::begin(l);
                auto l_it_end = std::end(l);
                while (l_it != l_it_end)
                    *(l_it++) = pow((*(t_it++)) - (*(p_it++)), 2) / GradSize;
                return l;
            }

            static GradTp grad(const GradTp &target, const GradTp &pred)
            {
                GradTp g;
                const auto *t_it = std::begin(target);
                const auto *p_it = std::begin(pred);
                auto g_it = std::begin(g);
                auto g_it_end = std::end(g);
                while (g_it != g_it_end)
                    *(g_it++) = (2.0 / GradSize) * ((*(p_it++)) - (*(t_it++)));
                return g;
            }
        };

        template <unsigned int GradSize>
        struct MAE : public LossBase<GradSize>
        {
            typedef typename LossBase<GradSize>::GradTp GradTp;
            GradTp operator()(const GradTp &output, const GradTp &target) const
            {
                this->grad = output;
                return output;
            }
        };

        template <unsigned int GradSize>
        struct Huber : public LossBase<GradSize>
        {
            typedef typename LossBase<GradSize>::GradTp GradTp;
            GradTp operator()(const GradTp &output, const GradTp &target) const
            {
                this->grad = output;
                return output;
            }
        };
    }
}

#endif /* NEURAL_NETWORK_FUNCTION_LOSS_HPP */