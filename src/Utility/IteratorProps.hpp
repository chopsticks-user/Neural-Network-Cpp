#ifndef UTILITY_ITERATOR_PROPS_HPP
#define UTILITY_ITERATOR_PROPS_HPP

#include <tuple>

namespace utility
{
    template <typename IteratorType, typename RowSizeTp>
    std::tuple<IteratorType, IteratorType, int>
    matrix_it_props(IteratorType it_begin, IteratorType it_end, RowSizeTp row_size)
    {
        return std::make_tuple(it_begin, it_end, row_size);
    }

    /**
     * @brief Iterator wrapper for matrix data strucures.
     *
     * @tparam ItType
     * @tparam SizeTp
     */
    template <typename ItTp, typename SizeTp>
    struct MatrixIt
    {
        ItTp begin;
        ItTp end;
        SizeTp row_size;

        MatrixIt(ItTp it_begin, ItTp it_end, SizeTp size)
        {
            expect((it_end - it_begin) % size == 0 && it_end - it_begin > 0,
                   std::runtime_error("Not a matrix."));
            begin = it_begin;
            end = it_end;
            row_size = size;
        };
    };

    /**
     * @brief Iterator wrapper for one-dimensional data strucures.
     *
     * @tparam ItType
     * @tparam RowSizeTp
     */
    template <typename ItTp>
    struct VectorIt
    {
        ItTp begin;
        ItTp end;

        VectorIt(ItTp it_begin, ItTp it_end)
            : begin(it_begin), end(it_end){};

        template <typename SizeTp>
        VectorIt(ItTp it_begin, SizeTp size)
            : begin(it_begin), end(it_begin + size){};
    };
}

#endif /* UTILITY_ITERATOR_PROPS_HPP */