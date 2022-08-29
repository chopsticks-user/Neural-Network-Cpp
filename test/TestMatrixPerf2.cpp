#include "src/LinearAlgebra.hpp"
#include "src/Exception/Exception.hpp"

#include <list>
#include <array>
#include <typeinfo>
#include <functional>
#include <string_view>

using namespace linear_algebra;
using namespace zz_no_inc;
using utility::Timer;

class A
{
public:
    int aa;
    int ac;
    int ad;
    A() { aa = 2; };
    A(const A &a) = delete;
    ~A() = default;
};

template <typename T>
constexpr bool test(T &&x)
{
    std::cout << std::addressof(x) << '\n';
    if constexpr (std::is_lvalue_reference_v<T>)
        return true;
    else
        return false;
}

// Matrix<ElementType> result(row_size, col_size);

// if (scalar == 0)
//     return result;

// auto it = this -> begin();
// auto it_end = this -> end();
// auto res_it = result.begin();
// auto res_it_end = result.end();

// SizeType i = 0;
// while ((res_it + i) != res_it_end)
// {
//     *(res_it + i) = (*(it + i)) * (scalar);
//     i++;
// }

// return result;

int main()
{

    try
    {
        int n = 1600;
        int size = n * n;
        {
            int sum = 0;
            Matrix<int> m1(n, n, 12);
            Matrix<int> m2(n, n, -2);

            {
                utility::Timer t1;
                m1 + m2;
                // m1 * m2;
            }
            // 10000 x 10000 elements
            // Addition/Subtraction
            // Old: Time elapsed: 9,114,394 μs
            // New: Time elapsed: 249,774 μs

            // Scalar multiplication
            // Old: Time elapsed: 3,062,789 μs
            // New: Time elapsed: 244,314 μs

            // 500x500 elements
            // Matrix multiplication
            // Old: Time elapsed: 11,792,090 μs
            // New: Time elapsed: 375,278 μs

            // 10000x10000 elements
            // Row addition
            // Old: Time elapsed: 161 μs
            // New: Time elapsed: 15 μs

            // 10000x10000 elements
            // Row swap
            // Old: Time elapsed: 325 μs
            // New: Time elapsed: 35 μs

            // 10000x10000 elements
            // Transpose
            // Old: Time elapsed: 6,761,776 μs
            // New: Time elapsed: 339,611 μs
        }

        {
            int sum = 0;
            std::vector<int> v1(size, -2383);
            utility::Timer t2;
            for (auto it = v1.begin(); it != v1.end(); ++it)
            {
                *it;
            }
            // for (auto i = 0; i < size; i++)
            // {
            //     v1[i];
            // }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
