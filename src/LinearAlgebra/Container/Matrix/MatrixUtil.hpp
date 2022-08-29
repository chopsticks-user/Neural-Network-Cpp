#ifndef LIN_ALG_CONTAINER_MATRIX_UTIL_HPP
#define LIN_ALG_CONTAINER_MATRIX_UTIL_HPP

#include <type_traits>
#include <iostream>

namespace zz_no_inc
{

    namespace matrix
    {
        typedef int_least64_t SizeType;

        template <SizeType col_size, SizeType row_size>
        constexpr void validate_matrix_dimensions()
        {
            static_assert(col_size >= 0,
                          "Row size must be non-negative.");
            static_assert(row_size >= 0,
                          "Column size must be non-negative.");
            static_assert(col_size != 0 || row_size == 0,
                          "Dynamic matrices cannot have a fixed column size.");
        }

        template <SizeType col_size, SizeType row_size>
        constexpr bool check_if_dynamic_matrix()
        {
            validate_matrix_dimensions<col_size, row_size>();
            return col_size == 0;
        }

        bool check_if_dynamic_matrix(SizeType col_size, SizeType row_size)
        {
            return col_size == 0 && row_size == 0;
        }

        template <SizeType col_size, SizeType row_size>
        constexpr bool check_if_static_matrix()
        {
            validate_matrix_dimensions<col_size, row_size>();
            return col_size > 0;
        }

        bool check_if_static_matrix(SizeType col_size, SizeType row_size)
        {
            return col_size > 0;
        }

        template <SizeType col_size, SizeType row_size>
        constexpr bool check_if_static_square_matrix()
        {
            validate_matrix_dimensions<col_size, row_size>();
            return col_size > 0 && (row_size == 0 || col_size == row_size);
        }

        bool check_if_static_square_matrix(SizeType col_size, SizeType row_size)
        {
            return col_size > 0 && (row_size == 0 || col_size == row_size);
        }

        template <SizeType col_size, SizeType row_size>
        constexpr SizeType verified_matrix_row_size()
        {
            if constexpr (check_if_static_square_matrix<col_size, row_size>() == true)
                return col_size;
            return row_size;
        }

        SizeType verified_matrix_row_size(SizeType col_size, SizeType row_size)
        {
            if (check_if_static_square_matrix(col_size, row_size) == true)
                return col_size;
            return row_size;
        }

        template <SizeType col_size, SizeType row_size>
        constexpr SizeType verified_matrix_data_container_size()
        {
            return col_size * verified_matrix_row_size<col_size, row_size>();
        }

        SizeType verified_matrix_data_container_size(
            SizeType col_size,
            SizeType row_size)
        {
            return col_size * verified_matrix_row_size(col_size, row_size);
        }

        template <SizeType col_size1, SizeType row_size1,
                  SizeType col_size2, SizeType row_size2>
        constexpr bool check_if_equal_dimensions()
        {
            if constexpr (col_size1 == col_size2)
                if constexpr (verified_matrix_row_size<col_size1, row_size1>() ==
                              verified_matrix_row_size<col_size2, row_size2>())
                    return true;
            return false;
        }

        bool check_if_equal_dimensions(SizeType col_size1, SizeType row_size1,
                                       SizeType col_size2, SizeType row_size2)
        {
            return col_size1 == col_size2 && row_size1 == row_size2;
        }

        bool check_if_multipliable(SizeType col_size1, SizeType row_size1,
                                   SizeType col_size2, SizeType row_size2)
        {
            return col_size1 == row_size2 && row_size1 == col_size2;
        }

        template <SizeType col_size, SizeType row_size>
        struct is_declared_dynamic_matrix
            : public std::integral_constant<
                  bool, check_if_dynamic_matrix<col_size, row_size>()>
        {
        };

        // template <SizeType col_size, SizeType row_size>
        // using is_declared_dynamic_matrix_v =
        //     typename is_declared_dynamic_matrix<col_size, row_size>::value;

        template <SizeType col_size, SizeType row_size>
        struct is_declared_static_matrix
            : public std::integral_constant<
                  bool, check_if_static_matrix<col_size, row_size>()>
        {
        };

        // template <SizeType col_size, SizeType row_size>
        // using is_declared_static_matrix_v =
        //     typename is_declared_static_matrix<col_size, row_size>::value;

        template <typename ContainerItType>
        void print_1d_container(ContainerItType it_begin, ContainerItType it_end)
        {
            std::cout << '[' << *it_begin;
            for (; it_begin != it_end; it_begin++)
                std::cout << ", " << *it_begin;
            std::cout << "]\n";
        }

        // in Matrix.hpp
        template <typename T>
        struct is_dynamic_matrix : public std::false_type
        {
        };

        // in Matrix.hpp
        template <template <typename ElementType,
                            SizeType col_size,
                            SizeType row_size>
                  class T,
                  typename ElementType>
        struct is_dynamic_matrix<T<ElementType, 0, 0>>
            : public std::true_type
        {
        };

        template <typename SeqContainer1D>
        SizeType get_1d_seq_container_size(const SeqContainer1D &container)
        {
            if constexpr (std::is_array_v<SeqContainer1D>)
                return sizeof(container) / sizeof(container[0]); // might throw
            else
                return container.size(); // might throw
        }

        template <typename ElementType>
        constexpr void check_element_requirements()
        {
            static_assert(std::is_default_constructible_v<ElementType>,
                          "Element type must be default constructible.");
            static_assert(std::is_copy_constructible_v<ElementType>,
                          "Element type must be copy constructible.");
            static_assert(std::is_move_constructible_v<ElementType>,
                          "Element type must be move constructible.");
            static_assert(std::is_nothrow_destructible_v<ElementType>,
                          "Element type must be non-throwing destructible.");
        }
    }

}

#endif /* LIN_ALG_CONTAINER_MATRIX_UTIL_HPP */