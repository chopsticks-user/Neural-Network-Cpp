#include <stdexcept>

namespace exception
{
    class OutOfRangeException : public std::out_of_range
    {

    };
}

#define ZZ_THROW_OUT_OF_RANGE_EXCEPTION__ throw std::out_of_range("")