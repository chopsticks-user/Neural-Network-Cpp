#ifndef LINEAR_ALGEBRA_COMPUTATION_MATRIX_HPP
#define LINEAR_ALGEBRA_COMPUTATION_MATRIX_HPP

#include "../Utility/Utility.hpp"

#include <math.h>

namespace linear_algebra
{
    namespace matrix
    {
        namespace helper
        {

        } /* helper */

        template <typename ItTp, typename RowSizeTp>
        void add(utility::MatrixIt<ItTp, RowSizeTp> it1,
                 utility::MatrixIt<ItTp, RowSizeTp> it2,
                 utility::MatrixIt<ItTp, RowSizeTp> it_res,
                 bool require_pre_cond = true)
        {
            if (require_pre_cond)
                utility::expect(it1.row_size == it2.row_size && it2.row_size == it_res.row_size &&
                                    it1.end - it1.begin == it2.end - it2.begin &&
                                    it1.end - it1.begin == it_res.end - it_res.begin,
                                std::range_error("Matrix addition: Dimensions mismatch."));
            RowSizeTp len = it_res.end - it_res.begin;
            for (RowSizeTp i = 0; i < len; ++i)
                *(it_res.begin + i) = *(it1.begin + i) + *(it2.begin + i);
        }

        template <typename ItTp, typename RowSizeTp>
        void subtract(utility::MatrixIt<ItTp, RowSizeTp> it1,
                      utility::MatrixIt<ItTp, RowSizeTp> it2,
                      utility::MatrixIt<ItTp, RowSizeTp> it_res,
                      bool require_pre_cond = true)
        {
            if (require_pre_cond)
                utility::expect(it1.row_size == it2.row_size && it2.row_size == it_res.row_size &&
                                    it1.end - it1.begin == it2.end - it2.begin &&
                                    it1.end - it1.begin == it_res.end - it_res.begin,
                                std::range_error("Matrix subtraction: Dimensions mismatch."));
            RowSizeTp len = it_res.end - it_res.begin;
            for (RowSizeTp i = 0; i < len; ++i)
                *(it_res.begin + i) = *(it1.begin + i) - *(it2.begin + i);
        }

        template <typename ItTp, typename SizeTp>
        void multiply(utility::MatrixIt<ItTp, SizeTp> it1,
                      utility::MatrixIt<ItTp, SizeTp> it2,
                      utility::MatrixIt<ItTp, SizeTp> it_res,
                      bool require_pre_cond = true)
        {
            SizeTp it1_col_h = (it1.end - it1.begin) / it1.row_size;
            SizeTp it1_row_w = it1.row_size;
            SizeTp it2_col_h = (it2.end - it2.begin) / it2.row_size;
            SizeTp it2_row_w = it2.row_size;
            SizeTp it_res_col_h = (it_res.end - it_res.begin) / it_res.row_size;
            SizeTp it_res_row_w = it_res.row_size;
            if (require_pre_cond)
                utility::expect(it1_row_w == it2_col_h &&
                                    it1_col_h == it_res_col_h &&
                                    it2_row_w == it_res_row_w,
                                std::range_error("Matrix multiplication: Dimensions mismatch."));
            for (SizeTp i = 0; i < it1_col_h; i++)
                for (SizeTp j = 0; j < it2_row_w; j++)
                    for (SizeTp k = 0; k < it2_col_h; k++)
                        *(it_res.begin + i * it2_row_w + j) +=
                            (*(it1.begin + i * it2_col_h + k)) *
                            (*(it2.begin + k * it2_row_w + j));
        }

        template <typename ItTp, typename RowSizeTp, typename ScalarTp>
        void scalar_multiply(utility::MatrixIt<ItTp, RowSizeTp> it1,
                             ScalarTp scalar,
                             utility::MatrixIt<ItTp, RowSizeTp> it_res,
                             bool require_pre_cond = true)
        {
            if (require_pre_cond)
                utility::expect(it1.row_size == it_res.row_size &&
                                    it1.end - it1.begin == it_res.end - it_res.begin,
                                std::range_error("Matrix-scalar multiplication: Dimensions mismatch."));
            RowSizeTp len = it_res.end - it_res.begin;
            for (RowSizeTp i = 0; i < len; ++i)
                *(it_res.begin + i) = *(it1.begin + i) * scalar;
        }
    } /* matrix */

} /* linear_algebra */

// template <typename ItTp, typename SizeTp, template <typename T, typename U> class... Its>
// auto add(utility::MatrixIt<ItTp, SizeTp> res, Its<ItTp, SizeTp>... args)
// {
//     static_assert((std::is_same_v<Its<ItTp, SizeTp>, utility::MatrixIt<ItTp, SizeTp>>, ...));
//     // ((std::cout << args.begin << '\n'), ...);
// }

// template <typename UnitTp, template <typename T> class... MatTp>
// void test(const MatTp<UnitTp> &...args)
// {
//     static_assert((std::is_same_v<MatTp<UnitTp>, Matrix<UnitTp>>, ...));
//     ((std::cout << args << '\n'), ...);
// }

// template <int I, class... Ts>
// decltype(auto) get(Ts &&...ts)
// {
//     return std::get<I>(std::forward_as_tuple(ts...));
// }

// template <typename... MatTp>
// void test1(const MatTp &...args)
// {
//     auto tuple = std::forward_as_tuple(args...);
//     // auto& first = get<0>(args...);
//     // ((std::cout << args.begin << '\n'), ...);
//     // std::cout << get<0>(args...);
// }

#endif /* LINEAR_ALGEBRA_COMPUTATION_MATRIX_HPP */