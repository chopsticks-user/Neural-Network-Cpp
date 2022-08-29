#include "src/LinearAlgebra/LinearAlgebra.hpp"

using namespace linear_algebra;
using utility::MatrixIt;
using utility::Timer;

int main()
{

    try
    {
        Timer t;
        Matrix<double> m1(100, 100);
        m1*m1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 0;
    }
    return 0;
}
