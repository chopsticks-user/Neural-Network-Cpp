#ifndef LIN_ALG_CONTAINER_DYNAMIC_MATRIX_HPP
#define LIN_ALG_CONTAINER_DYNAMIC_MATRIX_HPP

#include "../../Config.hpp"

#include "MatrixBase.hpp"

#include <queue>

namespace zz_no_inc
{
    template <typename ElementType>
    class DynamicMatrix_
        : public MatrixBase_<ElementType>
    {
        typedef MatrixBase_<ElementType> Base_;
        typedef matrix::SizeType SizeType;

    public:
        DynamicMatrix_()
        {
            this->set_dimensions_();
        }
#if DEBUG_LIN_ALG
        ~DynamicMatrix_() noexcept
        {
            std::cout << "At memory address: <" << std::addressof(*this)
                      << ">, an instance of <DynamicMatrix>, whose size of " << sizeof(*this)
                      << " bytes, has been destroyed.\n";
        }
#else
        ~DynamicMatrix_() = default;
#endif /* DEBUG_LIN_ALG */

        /// Might throw (std::bad_alloc) if std::fill_n failed to allocate memory.
        explicit DynamicMatrix_(SizeType col_size,
                                SizeType row_size = 0,
                                const ElementType &fill_value = ElementType())
        {
            this->fill_initialize_(col_size, row_size, fill_value);
        }

        DynamicMatrix_(const DynamicMatrix_ &rhs_matrix)
        {
            this->copy_initialize_(rhs_matrix);
        };

        template <typename RhsMatrixType,
                  SizeType col_size,
                  SizeType row_size>
        DynamicMatrix_(const MatrixBase_<RhsMatrixType, col_size, row_size>
                           &rhs_matrix)
        {
            this->copy_initialize_(rhs_matrix);
        }

        DynamicMatrix_(DynamicMatrix_ &&rhs_matrix) = default;

        template <typename RhsMatrixType,
                  SizeType col_size,
                  SizeType row_size>
        DynamicMatrix_(MatrixBase_<RhsMatrixType, col_size, row_size>
                           &&rhs_matrix)
        {
            this->move_initialize_(std::move(rhs_matrix));
        }

        template <typename SeqContainer1D>
        DynamicMatrix_ &insert_row_at_(SizeType row_index,
                                       const SeqContainer1D &rhs_container)
        {
            SizeType rhs_size = matrix::get_1d_seq_container_size(rhs_container);
            if (rhs_size != this->n_cols__)
                throw std::range_error("Row sizes mismatch.");
            if (row_index > this->n_rows__ || row_index < 0)
                throw std::range_error("Row index out of range.");

            this->data__.insert(this->data__.begin() + row_index * this->n_cols__,
                                std::begin(rhs_container),
                                std::end(rhs_container));
            this->n_rows__++;
            return *this;
        }

        template <typename SeqContainer1D>
        DynamicMatrix_ &insert_col_at_(SizeType col_index,
                                       const SeqContainer1D &rhs_container)
        {
            SizeType rhs_size = matrix::get_1d_seq_container_size(rhs_container); // might throw
            if (rhs_size != this->n_rows__)
                throw std::range_error("Column sizes mismatch.");
            if (col_index > this->n_cols__ || col_index < 0)
                throw std::range_error("Column index out of range.");

            // could be in an invalid state if an exception is thrown
            this->data__.resize(this->data__.size() + rhs_size);

            SizeType new_n_cols = this->n_cols__ + 1;
            std::queue<ElementType> q;
            for (int i = col_index, j = 0; i < this->n_rows__ * new_n_cols; i++)
            {
                q.push(this->data__[i]);
                if (i % new_n_cols == col_index)
                    this->data__[i] = rhs_container[j++];
                else
                {
                    this->data__[i] = q.front();
                    q.pop();
                }
            } // might throw

            this->n_cols__ = new_n_cols;
            return *this;
        };

        DynamicMatrix_ &erase_row_at_(SizeType row_index)
        {
            if (row_index >= this->n_rows__ || row_index < 0)
                throw std::range_error("Row index out of range.");
            this->data__.erase(this->data__.begin() + this->n_cols__ * row_index,
                               this->data__.begin() + this->n_cols__ * row_index + this->n_cols__);
            this->n_rows__--;
            return *this;
        };

        DynamicMatrix_ &erase_col_at_(SizeType col_index)
        {
            if (col_index >= this->n_cols__ || col_index < 0)
                throw std::range_error("Column index out of range.");

            SizeType new_n_cols = this->n_cols__ - 1;
            for (SizeType i = col_index, j = 0; i < this->n_rows__ * (new_n_cols) + col_index; i++)
            {
                if (i == new_n_cols * j + col_index)
                    j++;
                this->data__[i] = this->data__[i + j];
            }

            this->data__.resize(this->n_rows__ * new_n_cols);
            this->n_cols__ = new_n_cols;
            return *this;
        };

        // fill value = default value of ElementType, check if ElementType is default constructible
        DynamicMatrix_ &resize_and_fill_default_(SizeType new_n_rows,
                                                 SizeType new_n_cols,
                                                 ElementType fill_value = ElementType())
        {
            this->set_dimensions_(new_n_rows, new_n_cols);
            this->data__.resize(new_n_rows * new_n_cols);
            this->fill_all_element_with_(fill_value);
            return *this;
        }

        DynamicMatrix_ &resize_with_new_top_left_corner_(){};

        DynamicMatrix_ &clear_data_()
        {
            this->set_dimensions_(0, 0);
            this->data__.clear();
            return *this;
        };
    };
}

#endif /* LIN_ALG_CONTAINER_DYNAMIC_MATRIX_HPP */