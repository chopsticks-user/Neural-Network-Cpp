#ifndef UTILITY_EXPECT_HPP
#define UTILITY_EXPECT_HPP

#include <stdexcept>

namespace utility
{
    template <typename ExceptionType>
    void expect(bool condition,
                const ExceptionType &response_to_unexpected)
    {
        if (condition == true)
            return;
        throw response_to_unexpected;
    }

    template <bool condition, typename ExceptionType>
    void expect(const ExceptionType &response_to_unexpected)
    {
        return expect(condition, response_to_unexpected);
    }

} /* utility */

#endif /* UTILITY_EXPECT_HPP */