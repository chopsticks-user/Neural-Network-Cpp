#include "src/LinearAlgebra.hpp"
#include "src/Exception/Exception.hpp"

#include <list>
#include <array>
#include <typeinfo>

using namespace linear_algebra;

class A
{
public:
    int aa;
    int ac;
    int ad;
    A() { aa = 0; };
    A(const A &a) = delete;
    ~A() = default;
};

int main()
{

    try
    {
        utility::Timer *t1 = new utility::Timer();
        Matrix<int, 10000, 10000> m1(1);
        Matrix<int, 10000, 10000> m2;
        m2 = std::move(m1);
        delete t1;

        utility::Timer *t2 = new utility::Timer();
        std::vector<int> v1(100000000, 1);
        std::vector<int> v2;
        v2 = v1;
        delete t2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}