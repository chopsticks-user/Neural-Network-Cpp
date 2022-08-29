#ifndef LIN_ALG_CONTAINER_BASEMATRIX_HPP
#define LIN_ALG_CONTAINER_BASEMATRIX_HPP

#include "../../Config.hpp"

#include "MatrixUtil.hpp"
#include "../../Algorithm/Algorithm.hpp"
#include "../../Utility/Utility.hpp"

#include <vector>
#include <array>
#include <memory>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace zz_no_inc
{
    template <typename ElementType,
              matrix::SizeType templ_col_size = 0,
              matrix::SizeType templ_row_size = 0>
    struct MatrixBase_
    {
        typedef matrix::SizeType SizeType;

        typedef std::vector<ElementType> DataContainerType;

        template <typename ReturnType>
        using DynamicMatrixMethod = typename std::enable_if_t<
            matrix::is_declared_dynamic_matrix<templ_col_size,
                                                templ_row_size>{},
            ReturnType>;

        template <typename ReturnType>
        using StaticMatrixMethod = typename std::enable_if_t<
            matrix::is_declared_static_matrix<templ_col_size,
                                               templ_row_size>{},
            ReturnType>;

        /**
         * If both the size of column and row are either unspecified or equal to zero, data is stored
         * in a standard array. Otherwise, data is stored in a standard vector.
         */
        DataContainerType data__;
        SizeType n_rows__;
        SizeType n_cols__;

        constexpr bool is_dynamic_()
        {
            return matrix::is_declared_dynamic_matrix<templ_col_size,
                                                       templ_row_size>{};
        }

        constexpr bool is_static_()
        {
            return matrix::is_declared_static_matrix<templ_col_size,
                                                      templ_row_size>{};
        }

        const SizeType &get_col_size_() const noexcept { return n_rows__; }

        const SizeType &get_row_size_() const noexcept { return n_cols__; }

        template <typename ReturnType = void>
        StaticMatrixMethod<ReturnType>
        set_dimensions_()
        {
            matrix::check_element_requirements<ElementType>();
            n_rows__ = templ_col_size;
            n_cols__ = matrix::verified_matrix_row_size<templ_col_size, templ_row_size>();
        }

        template <typename ReturnType = void>
        DynamicMatrixMethod<ReturnType>
        set_dimensions_(SizeType col_size = 0, SizeType row_size = 0)
        {
            matrix::check_element_requirements<ElementType>();
            n_rows__ = col_size;
            n_cols__ = matrix::verified_matrix_row_size(col_size, row_size);
        }

        // fill value = default value of ElementType, check if ElementType is default constructible
        template <typename ReturnType = void>
        StaticMatrixMethod<ReturnType>
        fill_initialize_(ElementType fill_value = ElementType())
        {
            set_dimensions_();
            data__.resize(templ_col_size * templ_row_size);
            data__.shrink_to_fit();
        }

        // fill value = default value of ElementType, check if ElementType is default constructible
        template <typename ReturnType = void>
        DynamicMatrixMethod<ReturnType>
        fill_initialize_(SizeType col_size,
                         SizeType row_size,
                         ElementType fill_value = ElementType())
        {
            set_dimensions_(col_size, row_size);
            data__.resize(col_size * row_size, fill_value);
        }

        template <typename RhsElementType, SizeType rhs_col_size, SizeType rhs_row_size,
                  typename ReturnType = void>
        StaticMatrixMethod<ReturnType>
        copy_initialize_(const MatrixBase_<RhsElementType,
                                           rhs_col_size,
                                           rhs_row_size>
                             &rhs_matrix)
        {
            if constexpr (rhs_col_size != 0)
                static_assert(matrix::check_if_equal_dimensions<
                                  templ_col_size, templ_row_size,
                                  rhs_col_size, rhs_row_size>(),
                              "Copy to a static matrix: Dimensions mismatch.");

            set_dimensions_();
            if (rhs_matrix.n_rows__ != templ_col_size ||
                rhs_matrix.n_cols__ != matrix::verified_matrix_row_size(templ_col_size, templ_row_size))
                throw std::runtime_error("Copy to a static matrix: Dimensions mismatch.");
            data__ = rhs_matrix.data__;
            data__.shrink_to_fit();
        }

        template <typename RhsElementType, SizeType rhs_col_size, SizeType rhs_row_size,
                  typename ReturnType = void>
        DynamicMatrixMethod<ReturnType>
        copy_initialize_(const MatrixBase_<RhsElementType,
                                           rhs_col_size,
                                           rhs_row_size>
                             &rhs_matrix)
        {
            if constexpr (rhs_col_size != 0)
                set_dimensions_(rhs_col_size, rhs_row_size);
            else
                set_dimensions_(rhs_matrix.n_rows__, rhs_matrix.n_cols__);
            data__ = rhs_matrix.data__;
        }

        template <typename RhsElementType, SizeType rhs_col_size, SizeType rhs_row_size,
                  typename ReturnType = void>
        StaticMatrixMethod<ReturnType>
        move_initialize_(MatrixBase_<RhsElementType,
                                     rhs_col_size,
                                     rhs_row_size>
                             &&rhs_matrix)
        {
            if constexpr (rhs_col_size != 0)
                static_assert(matrix::check_if_equal_dimensions<
                                  templ_col_size, templ_row_size,
                                  rhs_col_size, rhs_row_size>(),
                              "Copy to a static matrix: Dimensions mismatch.");

            set_dimensions_();
            if (rhs_matrix.n_rows__ != templ_col_size ||
                rhs_matrix.n_cols__ != matrix::verified_matrix_row_size(templ_col_size, templ_row_size))
                throw std::runtime_error("Copy to a static matrix: Dimensions mismatch.");
            data__ = std::move(rhs_matrix.data__);
        }

        template <typename RhsElementType, SizeType rhs_col_size, SizeType rhs_row_size,
                  typename ReturnType = void>
        DynamicMatrixMethod<ReturnType>
        move_initialize_(MatrixBase_<RhsElementType,
                                     rhs_col_size,
                                     rhs_row_size>
                             &&rhs_matrix)
        {
            if constexpr (rhs_col_size != 0)
                set_dimensions_(rhs_col_size, rhs_row_size);
            // fill_initialize_(rhs_col_size, rhs_row_size);
            else
                set_dimensions_(rhs_matrix.n_rows__, rhs_matrix.n_cols__);
            // fill_initialize_(rhs_matrix.n_rows__, rhs_matrix.n_cols__);
            data__ = std::move(rhs_matrix.data__);
        }

        ElementType &operator()(SizeType row_index, SizeType col_index)
        {
#if ALLOW_NEGATIVE_INDEX
            return data__[row_index * n_cols__ + col_index];
#else
            utility::expect(row_index >= 0 && row_index < n_rows__ &&
                                col_index >= 0 && col_index < n_cols__,
                            std::out_of_range("Index out of range."));
            return data__[row_index * n_cols__ + col_index];
#endif /* ALLOW_NEGATIVE_INDEX */
        }

        const ElementType &operator()(SizeType row_index, SizeType col_index) const
        {
#if ALLOW_NEGATIVE_INDEX
            return data__[row_index * n_cols__ + col_index];
#else
            utility::expect(row_index >= 0 && row_index < n_rows__ &&
                                col_index >= 0 && col_index < n_cols__,
                            std::out_of_range("Index out of range."));
            return data__[row_index * n_cols__ + col_index];
#endif /* ALLOW_NEGATIVE_INDEX */
        }

        /// No-throw if ElementType can be printed by std::cout
        friend std::ostream &operator<<(std::ostream &os, const MatrixBase_ &rhs_matrix)
        {
            os << "\n[";
            for (size_t i = 0; i < rhs_matrix.n_rows__; i++)
            {
                os << "[";
                if (rhs_matrix.data__.size() / rhs_matrix.n_rows__ >= 1)
                    os << rhs_matrix.data__[i * rhs_matrix.n_cols__];
                for (size_t j = 1; j < rhs_matrix.n_cols__; j++)
                    os << ", " << rhs_matrix.data__[i * rhs_matrix.n_cols__ + j];
                os << "]";
                if (i < rhs_matrix.n_rows__ - 1)
                    os << "\n ";
            }
            os << "]\n";
            return os;
        }

        DataContainerType clone_data_() const
        {
            DataContainerType cloned_data;
            if constexpr (matrix::is_declared_dynamic_matrix<templ_col_size,
                                                              templ_row_size>{})
                cloned_data.resize(n_rows__ * n_cols__);
            std::copy(data__.begin(), data__.end(), cloned_data.begin());
            return cloned_data;
        }

        MatrixBase_ &fill_all_element_with_(ElementType fill_value)
        {
            std::fill(data__.begin(), data__.end(), fill_value);
            return *this;
        }

        MatrixBase_ &fill_value_to_row_(SizeType row_index, ElementType fill_value)
        {
            if (row_index >= n_rows__)
                throw std::out_of_range("Row index is out of range.");
            std::fill(data__.begin() + n_cols__ * row_index,
                      data__.begin() + n_cols__ * (row_index + 1),
                      fill_value);
            return *this;
        }

        MatrixBase_ &fill_value_to_col_(SizeType col_index, ElementType fill_value)
        {
            if (col_index >= n_cols__)
                throw std::out_of_range("Column index is out of range.");
            if (n_cols__ == 1)
                fill_all_element_with_(fill_value);
            for (auto i = col_index; i < n_rows__ * n_cols__; i += n_cols__)
                data__.at(i) = fill_value;
            return *this;
        }

        template <typename SeqContainer1D>
        MatrixBase_ &copy_data_to_row_(SizeType row_index,
                                       const SeqContainer1D &rhs_container)
        {
            SizeType rhs_size = matrix::get_1d_seq_container_size(rhs_container);
            if (rhs_size != n_cols__)
                throw std::out_of_range("Row sizes mismatch.");
            if (row_index >= n_rows__ || row_index < 0)
                throw std::out_of_range("Row index out of bounds.");

            std::copy(std::begin(rhs_container),
                      std::end(rhs_container),
                      data__.begin() + n_cols__ * row_index);
            return *this;
        }

        template <typename SeqContainer1D>
        MatrixBase_ &copy_data_to_col_(SizeType col_index,
                                       const SeqContainer1D &rhs_container)
        {
            SizeType rhs_size = matrix::get_1d_seq_container_size(rhs_container);
            if (rhs_size != n_rows__)
                throw std::out_of_range("Column sizes mismatch.");
            if (col_index >= n_cols__ || col_index < 0)
                throw std::out_of_range("Column index out of bounds.");

            for (SizeType i = 0; i < rhs_size; i++)
                data__[n_cols__ * i + col_index] = rhs_container[i];
            return *this;
        }
    };
} /* zz_no_inc */

#endif /* LIN_ALG_CONTAINER_BASEMATRIX_HPP */