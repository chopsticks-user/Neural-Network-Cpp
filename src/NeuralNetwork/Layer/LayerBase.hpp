#ifndef NEURAL_NETWORK_LAYERBASE_HPP
#define NEURAL_NETWORK_LAYERBASE_HPP

#include "../../LinearAlgebra/LinearAlgebra.hpp"

namespace neural_network
{
    typedef int_least64_t SizeTp;

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
    }
}

#endif /* NEURAL_NETWORK_LAYERBASE_HPP */