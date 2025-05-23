//
// Created by feder on 15/08/2024.
// lower_bound could not be placed in util.cuh for incomprehensible reasons
//

#ifndef HPC_PROJECT_2024_LOWER_BOUND_CUH
#define HPC_PROJECT_2024_LOWER_BOUND_CUH



/**
 * Searches for the first element x such that x <= value
 * @tparam T type of array element
 * @tparam Index integral type used to represent indexes
 * @param value value to search for
 * @param data array
 * @param length length of the array
 * @return index of the first element x such that x <= value
 * @see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 */
template<class T, class Index>
__device__ Index lower_bound(T value, const T * data, Index length) {
    #if CUDART_VERSION > 11000
    static_assert(cuda::std::is_integral<Index>::value, "Type used as index must be of integral type");
    #endif
    Index step, first = 0;
    while (length > 0) {
        step = length >> 1;
        if (data[first + step] < value) {
            first += step + 1;
            length -= step + 1;
        }
        else length = step;
    }
    return first;
}

/**
 * Searches for the first element x such that value < x
 * @tparam T type of array element
 * @tparam Index integral type used to represent indexes
 * @param value value to search for
 * @param data array
 * @param length m_length of the array
 * @return index of the first element x such that x <= value
 * @see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 */
template<class T, class Index>
__device__ Index upper_bound(T value, const T * data, Index length) {
//    static_assert(std::is_integral_v<Index>, "Type used as index must be of integral type");
    Index step, index, first = 0;
    while (length > 0)
    {
        index = first;
        step = length >> 1;
        index += step;
        if (data[index] <= value)
        {
            first = index + 1;
            length -= step + 1;
        }
        else length = step;
    }

    return first;
}

#endif //HPC_PROJECT_2024_LOWER_BOUND_CUH
