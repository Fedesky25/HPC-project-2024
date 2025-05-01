//
// Created by feder on 10/10/2024.
//

#ifndef HPC_PROJECT_2024_SORTER_CUH
#define HPC_PROJECT_2024_SORTER_CUH

#if CUDART_VERSION >= 11000
    #include "cub/device/device_radix_sort.cuh"
    #define VALUES_0 m_values[0]
    #define KEYS_0 m_keys[0]
    #define VALUES_CURRENT m_values[current]
    #define KEYS_CURRENT m_keys[current]
#else
    #include "thrust/sort.h"
    #define VALUES_0 m_values
    #define KEYS_0 m_keys
    #define VALUES_CURRENT m_values
    #define KEYS_CURRENT m_keys
#endif


template<class KeyType, class ValueType>
class KVSorter {

public:
    explicit KVSorter(size_t len)
    : m_length(len), use_buffer(false) {
        cudaMalloc(&VALUES_0, len * sizeof(ValueType));
        init();
    }
    KVSorter(size_t len, ValueType * values_buffer)
    : m_length(len), use_buffer(true) {
        VALUES_0 = values_buffer;
        init();
    }
    ~KVSorter() {
        cudaFree(KEYS_0);
        if(!use_buffer) cudaFree(VALUES_0);
        #if CUDART_VERSION > 11000
        else if(current) cudaMemcpy(m_values[0], m_values[1], m_length * sizeof(ValueType), cudaMemcpyDeviceToDevice);
        cudaFree(m_keys[1]);
        cudaFree(m_values[1]);
        cudaFree(temp_storage);
        #endif
    }
    inline size_t length() const { return m_length; }
    inline KeyType * keys() const { return KEYS_CURRENT; }
    inline ValueType * values() const { return VALUES_CURRENT; }

    void sort() {
        #if CUDART_VERSION >= 11000
        cub::DeviceRadixSort::SortPairs(
                temp_storage, temp_storage_bytes,
                m_keys[current], m_keys[1 ^ current],
                m_values[current], m_values[1 ^ current],
                m_length);
        current ^= 1;
        #else
        thrust::sort_by_key(thrust::device, m_keys, m_keys+m_length, m_values);
        #endif
    }

private:

    size_t m_length;
    #if CUDART_VERSION >= 11000
    size_t temp_storage_bytes;
    KeyType * m_keys[2];
    ValueType * m_values[2];
    void * temp_storage;
    char current = 0;
    #else
    KeyType * m_keys;
    ValueType * m_values;
    #endif
    bool use_buffer = false;

    void init() {
        CATCH_CUDA_ERROR(cudaMalloc(&KEYS_0, m_length * sizeof(KeyType)))
        #if CUDART_VERSION >= 11000
        CATCH_CUDA_ERROR(cudaMalloc(&m_keys[1], m_length * sizeof(KeyType)))
        CATCH_CUDA_ERROR(cudaMalloc(&m_values[1], m_length * sizeof(ValueType)))
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, m_keys[0], m_keys[1], m_values[0], m_values[1], m_length);
        CATCH_CUDA_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes))
        #endif
    }
};

#endif //HPC_PROJECT_2024_SORTER_CUH
