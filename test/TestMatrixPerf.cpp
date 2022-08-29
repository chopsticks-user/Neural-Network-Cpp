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

int main()
{

    try
    {
        int n = 200;
        int size = n * n;
        {
            int sum = 0;
            Matrix<int> m1(n, n, 1323);
            Matrix<int> m2(n, n, -2223);

            {
                utility::Timer t1;

                auto it1 = m1.begin();
                auto end1 = m1.end();
                auto ptr1 = &(*it1);
                auto ptr_end1 = &(*end1);

                auto it2 = m2.begin();
                auto end2 = m2.end();
                auto ptr2 = &(*it2);
                auto ptr_end2 = &(*end2);

                Matrix<int> m3(n, n, 0);

                auto it3 = m3.begin();
                auto end3 = m3.end();
                auto ptr3 = &(*it3);
                auto ptr_end3 = &(*end3);

                auto i = -1;

                while (ptr3 + (i++) != ptr_end3)
                {
                    *(ptr3 + i) = *(ptr1 + i) + *(ptr2 + i);
                }
                std::cout << *ptr3 << '\n';
            }

            {
                utility::Timer t1;
                std::cout << (m1 + m2)(0, 0) << '\n';
            }
        }

        {
            int sum = 0;
            std::vector<int> v1(size, -2383);
            std::vector<int> v2(size, 1231);
            utility::Timer t2;
            std::vector<int> v3(size, 0);
            // for (auto it = v1.begin(); it != v1.end(); ++it)
            // {
            //     sum += *it;
            //     if (sum & 1)
            //         *it += (*it) * 92821 + (*it);
            // }
            for (auto i = 0; i < size; i++)
            {
                v3[i] += v1[i] + v2[i];
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
