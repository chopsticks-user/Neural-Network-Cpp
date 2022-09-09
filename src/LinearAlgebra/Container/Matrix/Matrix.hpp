#ifndef LIN_ALG_CONTAINER_MATRIX_HPP
#define LIN_ALG_CONTAINER_MATRIX_HPP

#include "../../Config.hpp"

#include "../../Computation/Computation.hpp"
#include "DynamicMatrix.hpp"
#include "StaticMatrix.hpp"

#include <memory>
#include <cmath>

namespace linear_algebra
{
    template <typename UnitTp = float,
              zz_no_inc::matrix::SizeType tpl_col_h = 0,
              zz_no_inc::matrix::SizeType tpl_row_w = 0>
    class Matrix
    {
        typedef zz_no_inc::matrix::SizeType SizeTp;
        typedef zz_no_inc::StaticMatrix_<UnitTp,
                                         tpl_col_h,
                                         tpl_row_w>
            StcMatTp;

        typedef zz_no_inc::DynamicMatrix_<UnitTp> DycMatTp;

        template <typename ReturnTp>
        using DycMatMethod = typename std::enable_if_t<
            zz_no_inc::matrix::is_declared_dynamic_matrix<tpl_col_h,
                                                          tpl_row_w>{},
            ReturnTp>;

        template <typename ReturnTp>
        using StcMatMethod = typename std::enable_if_t<
            zz_no_inc::matrix::is_declared_static_matrix<tpl_col_h,
                                                         tpl_row_w>{},
            ReturnTp>;

        typedef std::conditional_t<zz_no_inc::matrix::check_if_dynamic_matrix<
                                       tpl_col_h,
                                       tpl_row_w>(),
                                   DycMatTp,
                                   StcMatTp>
            MatTp;

    public:
        auto copy_data() const
        {
            return *(this->mat_ptr_.get());
        }

        const auto &data() const
        {
            return *(this->mat_ptr_);
        }

        auto &data()
        {
            return *(this->mat_ptr_);
        }

        Matrix() : mat_ptr_(std::make_unique<MatTp>()){};

#if DEBUG_LIN_ALG
        ~Matrix() noexcept
        {
            std::cout << "At memory address: <" << std::addressof(*this)
                      << ">, an instance of <Matrix>, whose size of " << sizeof(*this)
                      << " bytes and contains a pointer at <" << this->mat_ptr_.get()
                      << ">, has been destroyed.\n";
        }
#else
        ~Matrix() = default;
#endif

        explicit Matrix(const UnitTp &fill_val)
            : mat_ptr_(std::make_unique<MatTp>(fill_val)){};

        explicit Matrix(SizeTp n_rows, SizeTp n_cols,
                        const UnitTp &fill_val = UnitTp())
            : mat_ptr_(std::make_unique<MatTp>(n_rows, n_cols, fill_val)){};

        Matrix(const Matrix &r_mat)
            : mat_ptr_(std::make_unique<MatTp>(std::move(r_mat.copy_data()))){};

        template <typename RUnitTp, SizeTp r_col_h, SizeTp r_row_w>
        Matrix(const Matrix<RUnitTp, r_col_h, r_row_w> &r_mat)
            : mat_ptr_(std::make_unique<MatTp>(r_mat.copy_data())){};

        Matrix(Matrix &&r_mat) = default;

        template <typename RUnitTp, SizeTp r_col_h, SizeTp r_row_w>
        Matrix(Matrix<RUnitTp, r_col_h, r_row_w> &&r_mat)
            : mat_ptr_(std::make_unique<MatTp>(std::move(r_mat.data()))){};

        Matrix &operator=(const Matrix &r_mat)
        {
            this->mat_ptr_ = std::make_unique<MatTp>(r_mat.copy_data());
            return *this;
        }

        template <typename RUnitTp, SizeTp r_col_h, SizeTp r_row_w>
        Matrix &operator=(const Matrix<RUnitTp, r_col_h, r_row_w> &r_mat)
        {
            this->mat_ptr_ = std::make_unique<MatTp>(r_mat.copy_data());
            return *this;
        }

        Matrix &operator=(Matrix &&r_mat) = default;

        template <typename RUnitTp, SizeTp r_col_h, SizeTp r_row_w>
        Matrix &operator=(Matrix<RUnitTp, r_col_h, r_row_w> &&r_mat)
        {
            this->mat_ptr_ = std::make_unique<MatTp>(std::move(r_mat.data()));
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &os, const Matrix &r_mat)
        {
            os << *(r_mat.mat_ptr_);
            return os;
        }

        UnitTp &operator()(SizeTp row_idx, SizeTp col_idx)
        {
#if ALLOW_NEGATIVE_INDEX
            return (*(this->mat_ptr_))(
                this->valid_row_index_(row_idx),
                this->valid_col_index(col_idx));
#else
            return (*(this->mat_ptr_))(row_idx, col_idx);
#endif /* ALLOW_NEGATIVE_INDEX */
        }

        const UnitTp &operator()(SizeTp row_idx, SizeTp col_idx) const
        {
            return (*(this->mat_ptr_))(row_idx, col_idx);
        }

        constexpr bool is_dynamic() noexcept
        {
            return zz_no_inc::matrix::is_declared_dynamic_matrix<tpl_col_h, tpl_row_w>{};
        }

        constexpr bool is_static() noexcept
        {
            return zz_no_inc::matrix::is_declared_static_matrix<tpl_col_h, tpl_row_w>{};
        }

        template <typename ReturnTp = SizeTp>
        constexpr StcMatMethod<ReturnTp>
        row_size() const noexcept
        {
            return tpl_row_w;
        }

        template <typename ReturnTp = SizeTp>
        DycMatMethod<ReturnTp> row_size() const noexcept
        {
            return this->mat_ptr_->get_row_size_();
        }

        template <typename ReturnTp = SizeTp>
        constexpr StcMatMethod<ReturnTp> column_size() const noexcept
        {
            return tpl_col_h;
        }

        template <typename ReturnTp = SizeTp>
        DycMatMethod<ReturnTp> column_size() const noexcept
        {
            return this->mat_ptr_->get_col_size_();
        }

        template <typename ReturnTp = bool>
        constexpr StcMatMethod<ReturnTp> is_square() const noexcept
        {
            return tpl_col_h == tpl_row_w;
        }

        template <typename ReturnTp = bool>
        DycMatMethod<ReturnTp> is_square() const noexcept
        {
            return this->row_size() == this->column_size();
        }

        auto clone_data() const
        {
            auto copy_ptr = std::make_unique<MatTp>();
            copy_ptr->data__ = std::move(this->mat_ptr_->clone_data_());
            if constexpr (zz_no_inc::matrix::is_declared_dynamic_matrix<tpl_col_h,
                                                                        tpl_row_w>{})
                copy_ptr->set_dimensions_(this->row_size(), this->column_size());
            return std::move(copy_ptr);
        }

        auto clone()
        {
            return Matrix<UnitTp, tpl_col_h, tpl_row_w>(*this);
        }

        auto begin() const noexcept { return &*(this->mat_ptr_->data__.begin()); };

        auto end() const noexcept { return &*(this->mat_ptr_->data__.end()); };

        auto it() const
        {
            return utility::MatrixIt(this->begin(), this->end(), this->row_size());
        }

        Matrix &fill(const UnitTp &fill_val = UnitTp())
        {
            this->mat_ptr_->fill_all_element_with_(fill_val);
            return *this;
        }

        Matrix& fill_random(UnitTp min, UnitTp max)
        {
            utility::rand(min, max, this->begin(), this->end());
            return *this;
        }

        Matrix &fill_row(SizeTp row_idx, const UnitTp &fill_val)
        {
            row_idx = this->valid_row_index_(row_idx);
            this->mat_ptr_->fill_val_to_row_(row_idx, fill_val);
            return *this;
        }

        template <typename SeqCtn1D>
        Matrix &fill_row(SizeTp row_idx, const SeqCtn1D &r_seq_ctn)
        {
            row_idx = this->valid_row_index_(row_idx);
            this->mat_ptr_->copy_data_to_row_(row_idx, r_seq_ctn);
            return *this;
        }

        Matrix &fill_column(SizeTp col_idx, const UnitTp &fill_val)
        {
            col_idx = this->valid_col_index(col_idx);
            this->mat_ptr_->fill_value_to_col_(col_idx, fill_val);
            return *this;
        }

        template <typename SeqCtn1D>
        Matrix &fill_column(SizeTp col_idx, const SeqCtn1D &r_seq_ctn)
        {
            col_idx = this->valid_col_index(col_idx);
            this->mat_ptr_->copy_data_to_col_(col_idx, r_seq_ctn);
            return *this;
        }

        template <typename SeqCtn1D, typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> append_row(SizeTp row_idx, const SeqCtn1D &r_seq_ctn)
        {
            row_idx = this->validate_negative_append_row_index_(row_idx);
            this->mat_ptr_->insert_row_at_(row_idx, r_seq_ctn);
            return *this;
        }

        template <typename SeqCtn1D, typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> append_column(SizeTp col_idx, const SeqCtn1D &r_seq_ctn)
        {
            col_idx = this->validate_negative_append_col_index_(col_idx);
            this->mat_ptr_->insert_col_at_(col_idx, r_seq_ctn);
            return *this;
        }

        template <typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> remove_row(SizeTp row_idx)
        {
            row_idx = this->valid_row_index_(row_idx);
            this->mat_ptr_->erase_row_at_(row_idx);
            return *this;
        }

        template <typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> remove_column(SizeTp col_idx)
        {
            col_idx = this->valid_col_index(col_idx);
            this->mat_ptr_->erase_col_at_(col_idx);
            return *this;
        }

        template <typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> resize(SizeTp new_col_size, SizeTp new_row_size)
        {
            this->mat_ptr_->resize_and_fill_default_(new_col_size, new_row_size);
            return *this;
        }

        template <typename ReturnTp = Matrix &>
        DycMatMethod<ReturnTp> clear()
        {
            this->mat_ptr_->clear_data_();
            return *this;
        }

#if ENABLE_MATRIX_MATH_FUNCTIONS
        /// UnitTp must have a defined operator+
        template <typename RMatTp>
        Matrix<UnitTp> operator+(const RMatTp &r_mat) const
        {
            const SizeTp r_col_h = r_mat.column_size();
            const SizeTp r_row_w = r_mat.row_size();

            utility::expect(this->is_summable(r_col_h, r_row_w),
                            std::runtime_error("Dimensions mismatch when performing matrix addition."));

            Matrix<UnitTp> result(r_col_h, r_row_w);
            matrix::add(this->it(), r_mat.it(), result.it(), false);
            return result;
        }

        /// UnitTp must have a defined operator-
        template <typename RMatTp>
        Matrix<UnitTp> operator-(const RMatTp &r_mat) const
        {
            const SizeTp r_col_h = r_mat.column_size();
            const SizeTp r_row_w = r_mat.row_size();

            utility::expect(this->is_summable(r_col_h, r_row_w),
                            std::runtime_error("Dimensions mismatch when performing matrix addition."));

            Matrix<UnitTp> result(r_col_h, r_row_w);
            matrix::subtract(this->it(), r_mat.it(), result.it(), false);
            return result;
        }

        /// UnitTp must have a defined operator*
        template <typename RMatTp>
        Matrix<UnitTp> operator*(const RMatTp &r_mat) const
        {
            const SizeTp col_h = this->column_size();
            const SizeTp r_col_h = r_mat.column_size();
            const SizeTp r_row_w = r_mat.row_size();

            utility::expect(
                this->is_multipliable(r_col_h),
                std::runtime_error(
                    "Dimensions mismatch when performing matrix multiplication."));

            Matrix<UnitTp> result(col_h, r_row_w);
            auto it1 = this->begin();
            auto it2 = r_mat.begin();
            auto res_it = result.begin();

            for (SizeTp i = 0; i < col_h; i++)
                for (SizeTp j = 0; j < r_row_w; j++)
                    for (SizeTp k = 0; k < r_col_h; k++)
                        *(res_it + i * r_row_w + j) +=
                            (*(it1 + i * r_col_h + k)) *
                            (*(it2 + k * r_row_w + j));

            return result;
        }

        /// UnitTp must have a defined operator*
        /// Old implementation: Time elapsed: 242,902 μs
        /// New implementation: Time elapsed: 217,647 μs
        Matrix<UnitTp> operator*(const UnitTp &scalar) const
        {

            static_assert(std::is_arithmetic_v<UnitTp>,
                          "UnitTp must be arithmetic type.");

            if (scalar == 0)
                return Matrix<UnitTp>(this->column_size(), this->row_size());
            if (scalar == 1)
                return *this;
            Matrix<UnitTp> result(this->column_size(), this->row_size());
            matrix::scalar_multiply(this->it(), scalar, result.it(), false);
            return result;
        }

        /// 10000x10000 elements: Time elapsed: 26,176 μs
        template <typename RMatTp>
        bool operator==(const RMatTp &r_mat)
        {
            return std::equal(this->begin(), this->end(), r_mat.begin(), r_mat.end());
        }

        template <typename RMatTp>
        bool operator!=(const RMatTp &r_mat)
        {
            return !std::equal(this->begin(), this->end(), r_mat.begin(), r_mat.end());
        }

        /// incomplete
        bool is_singular()
        {
            if (this->is_square() == false)
                return false;
            return true;
        }

        /// incomplete
        bool is_invertible()
        {
            return !this->is_singular();
        }

        /// 10000x10000 elements: 330,696 μs
        bool is_symmetric()
        {
            if (this->is_square() == false)
                return false;
            return (*this) == this->transpose();
        }

        /// 10000x10000 elements: 580,592 μs
        bool is_skew_symmetric()
        {
            if (this->is_square() == false)
                return false;
            return (*this) == this->transpose() * (-1);
        }

        /// incomplete
        bool is_definite()
        {
            if (this->is_symmetric() == false)
                return false;
            return true;
        }

        /// incomplete
        bool is_orthogonal()
        {
            if (this->is_square() == false)
                return false;
            return true;
        }

        /// 500x500 elements: Time elapsed: 162833 μs
        UnitTp det()
        {
            static_assert(std::is_arithmetic_v<UnitTp>,
                          "Scalar must be of any arithmetic type.");

            utility::expect(this->is_square() == true,
                            std::runtime_error(
                                "Taking power of a non-square matrix is not allowed."));

            SizeTp size = this->column_size();
            if (size == 0)
                return UnitTp();
            if (size == 1)
                return (*this)(0, 0);
            if (size == 2)
                return (*this)(0, 0) * (*this)(1, 1) - (*this)(1, 0) * (*this)(0, 1);

            auto copy_mat = this->clone();
            auto it = copy_mat.begin();
            UnitTp res = 1, total = 1;
            for (SizeTp i = 0; i < size; ++i)
            {
                SizeTp temp = i;
                while (temp < size)
                {
                    if (*(it + temp * size + i) != 0)
                        break;
                    ++temp;
                }
                if (temp == size)
                    continue;
                if (temp != i)
                {
                    copy_mat.row_swap(temp, i);
                    res *= std::pow(-1, temp - i);
                }
                for (SizeTp j = i + 1; j < size; ++j)
                {
                    UnitTp diag_val = *(it + i * size + i);
                    UnitTp next_row_val = *(it + j * size + i);
                    for (SizeTp k = 0; k < size; k++)
                        *(it + j * size + k) = (diag_val * (*(it + j * size + k))) -
                                               (next_row_val * (*(it + i * size + k)));
                    total *= diag_val;
                }
            }
            for (SizeTp i = 0; i < size; ++i)
                res *= (*(it + i * size + i));
            return res / total;
        }

        /// incomplete
        UnitTp norm() { return 0; }

        UnitTp trace()
        {
            utility::expect(this->is_square(),
                            std::runtime_error("Cannot find trace of a non-square matrix."));
            auto it = this->begin();
            auto size = this->column_size();
            UnitTp res = UnitTp();
            for (SizeTp i = 0; i < size; ++i)
                res += (*(it + i * size + i));
            return res;
        }

        Matrix identity()
        {
            /// exception: std::complex
            static_assert(std::is_arithmetic_v<UnitTp>,
                          "UnitTp must be of any arithmetic type.");

            utility::expect(this->is_square() == true,
                            std::runtime_error(
                                "Cannot find the identity matrix of a non-square matrix."));

            SizeTp col_size = this->column_size();
            Matrix<UnitTp> identity_matrix(col_size, col_size);
            auto it = identity_matrix.begin();

            for (SizeTp i = 0; i < col_size; ++i)
                *(it + i * (col_size + 1)) = 1;

            return identity_matrix;
        }

        /// 10000x10000 elements: Time elapsed: 102,164 μs
        Matrix diagonal() const
        {
            utility::expect(this->is_square() == true,
                            std::runtime_error("Cannot find the diagonal matrix of a non-square matrix."));
            SizeTp col_size = this->column_size();
            Matrix<UnitTp> res(col_size, col_size);
            auto it = this->begin();
            auto res_it = res.begin();
            for (SizeTp i = 0; i < col_size; ++i)
                *(res_it + i * (col_size + 1)) = *(it + i * (col_size + 1));
            return res;
        }

        /// 10000x10000 elements: Time elapsed: 186,823 μs
        Matrix lower_triangle()
        {
            utility::expect(this->is_square() == true,
                            std::runtime_error("Cannot find the diagonal matrix of a non-square matrix."));
            SizeTp col_size = this->column_size();
            Matrix<UnitTp> res(col_size, col_size);
            auto it = this->begin();
            auto res_it = res.begin();
            for (SizeTp i = 0; i < col_size; ++i)
                for (SizeTp j = 0; j <= i; ++j)
                    *(res_it + i * col_size + j) = *(it + i * col_size + j);
            return res;
        }

        /// 10000x10000 elements: Time elapsed: 189,699 μs
        Matrix upper_triangle()
        {
            utility::expect(this->is_square() == true,
                            std::runtime_error("Cannot find the diagonal matrix of a non-square matrix."));
            SizeTp col_size = this->column_size();
            Matrix<UnitTp> res(col_size, col_size);
            auto it = this->begin();
            auto res_it = res.begin();
            for (SizeTp i = 0; i < col_size; ++i)
                for (SizeTp j = i; j < col_size; ++j)
                    *(res_it + i * col_size + j) = *(it + i * col_size + j);
            return res;
        }

        /// incomplete
        Matrix inverse()
        {
            return *this;
        }

        /// 10000x10000 elements: Time elapsed: 418,555 μs
        Matrix sub(SizeTp rm_row_index, SizeTp rm_col_index)
        {
            Matrix<UnitTp> sub_matrix(std::move(this->clone()));
            sub_matrix.remove_row(rm_row_index);
            sub_matrix.remove_column(rm_col_index);
            return sub_matrix;
        }

        /// For testing purpose only. Actual algorithm does not repeatedly perform multiplication.
        template <typename ScalarType>
        Matrix pow(ScalarType scalar)
        {
            static_assert(std::is_arithmetic_v<ScalarType>,
                          "Scalar must be of any arithmetic type.");

            utility::expect(this->is_square() == true,
                            std::runtime_error(
                                "Taking power of a non-square matrix is not allowed."));

            if (scalar == 0)
                return this->identity();

            if (scalar == 1)
                return *this;

            if (scalar == -1)
                return this->inverse();

            const SizeTp col_size = this->column_size();
            const SizeTp row_size = this->row_size();
            Matrix<UnitTp> result(*this);

            for (SizeTp i = 0; i < scalar - 1; i++)
                result = std::move(result * result);
            return result;
        }

        /// Double-transposing on one matrix is not available.
        Matrix<UnitTp> transpose()
        {
            SizeTp t_col_size = this->row_size();
            SizeTp t_row_size = this->column_size();

            Matrix<UnitTp> result(t_col_size, t_row_size);
            auto it = this->begin();
            auto res_it = result.begin();

            for (SizeTp i = 0; i < t_col_size; ++i)
                for (SizeTp j = 0; j < t_row_size; ++j)
                    *(res_it + i * t_row_size + j) = *(it + j * t_col_size + i);
            return result;
        }

        template <typename ScalarType>
        Matrix &row_addition(SizeTp row_idx, ScalarType val)
        {
            static_assert(std::is_arithmetic_v<ScalarType>,
                          "Scalar must be of any arithmetic type.");

            row_idx = this->valid_row_index_(row_idx);
            SizeTp row_size = this->row_size();
            auto it = this->begin() + row_idx * row_size;
            auto it_end = this->begin() + row_idx * row_size + row_size;

            while (it != it_end)
                *(it++) += val;

            return *this;
        }

        template <typename ScalarType>
        Matrix &row_multiplication(SizeTp row_idx, ScalarType val)
        {
            static_assert(std::is_arithmetic_v<ScalarType>,
                          "Scalar must be of any arithmetic type.");

            row_idx = this->valid_row_index_(row_idx);
            SizeTp row_size = this->row_size();
            auto it = this->begin() + row_idx * row_size;
            auto it_end = this->begin() + row_idx * row_size + row_size;

            while (it != it_end)
                (*(it++)) *= val;

            return *this;
        }

        Matrix &row_swap(SizeTp row_idx1, SizeTp row_idx2)
        {
            row_idx1 = this->valid_row_index_(row_idx1);
            row_idx2 = this->valid_row_index_(row_idx2);

            SizeTp row_size = this->row_size();
            auto it1 = this->begin() + row_idx1 * row_size;
            auto it2 = this->begin() + row_idx2 * row_size;

            for (SizeTp i = 0; i < row_size; i++)
                std::swap(*(it1 + i), *(it2 + i));

            return *this;
        }
#endif /* MATRIX_MATH_FUNCTIONS */

    private:
        std::unique_ptr<MatTp> mat_ptr_;

        /// Need to check if the returned row_idx is still less than 0.
        SizeTp valid_row_index_(SizeTp row_idx) const
        {
#if ALLOW_NEGATIVE_INDEX
            if (row_idx < 0)
                row_idx += this->row_size();
#endif /* ALLOW_NEGATIVE_INDEX */
            utility::expect(row_idx >= 0 && row_idx < this->row_size(),
                            std::range_error("Row index out of range."));
            return row_idx;
        }

        SizeTp valid_col_index(SizeTp col_idx) const
        {
#if ALLOW_NEGATIVE_INDEX
            if (col_idx < 0)
                col_idx += this->column_size();
#endif /* ALLOW_NEGATIVE_INDEX */
            utility::expect(col_idx >= 0 && col_idx < this->column_size(),
                            std::range_error("Column index out of range."));
            return col_idx;
        }

        SizeTp validate_negative_append_row_index_(SizeTp row_idx) const
        {
            return row_idx < 0 ? this->column_size() + row_idx + 1 : row_idx;
        }

        SizeTp validate_negative_append_col_index_(SizeTp col_idx) const
        {
            return col_idx < 0 ? this->row_size() + col_idx + 1 : col_idx;
        }

        bool is_multipliable(SizeTp r_col_h) const noexcept
        {
            return this->row_size() == r_col_h;
        }

        bool is_summable(SizeTp r_col_h, SizeTp r_row_w) const noexcept
        {
            return zz_no_inc::matrix::check_if_equal_dimensions(
                this->column_size(), this->row_size(),
                r_col_h, r_row_w);
        }
    };
}

#endif /* LIN_ALG_CONTAINER_MATRIX_HPP */