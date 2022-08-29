#ifndef LIN_ALG_CONTAINER_STATIC_MATRIX_HPP
#define LIN_ALG_CONTAINER_STATIC_MATRIX_HPP

#include "../../Config.hpp"

#include "MatrixBase.hpp"

namespace zz_no_inc
{
    template <typename ElementType,
              matrix::SizeType templ_col_size = 0,
              matrix::SizeType templ_row_size = 0>
    class StaticMatrix_
        : public MatrixBase_<ElementType, templ_col_size, templ_row_size>
    {
        typedef MatrixBase_<ElementType, templ_col_size, templ_row_size> Base_;
        typedef matrix::SizeType SizeType;

    public:
        StaticMatrix_()
        {
            this->fill_initialize_();
        }

#if DEBUG_LIN_ALG
        ~StaticMatrix_() noexcept
        {
            std::cout << "At memory address: <" << std::addressof(*this)
                      << ">, an instance of <StaticMatrix>, whose size of " << sizeof(*this)
                      << " bytes, has been destroyed.\n";
        }
#else
        ~StaticMatrix_() = default;
#endif /* DEBUG_LIN_ALG */

        /// Might throw (std::bad_alloc) if std::fill_n failed to allocate memory.
        explicit StaticMatrix_(const ElementType &fill_value)
        {
            this->fill_initialize_(fill_value);
        }

        StaticMatrix_(const StaticMatrix_ &rhs_matrix)
        {
            this->copy_initialize_(rhs_matrix);
        };

        template <typename RhsMatrixType,
                  SizeType col_size,
                  SizeType row_size>
        StaticMatrix_(const MatrixBase_<RhsMatrixType, col_size, row_size>
                          &rhs_matrix)
        {
            this->copy_initialize_(rhs_matrix);
        };

        template <typename RhsMatrixType,
                  SizeType col_size,
                  SizeType row_size>
        StaticMatrix_(MatrixBase_<RhsMatrixType, col_size, row_size>
                          &&rhs_matrix)
        {
            this->move_initialize_(std::move(rhs_matrix));
        };

        StaticMatrix_(StaticMatrix_ &&rhs_matrix)
        {
            this->move_initialize_(std::move(rhs_matrix));
        }
    };
}

#endif /* LIN_ALG_CONTAINER_STATIC_MATRIX_HPP */