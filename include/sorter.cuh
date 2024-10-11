//
// Created by feder on 10/10/2024.
//

#ifndef HPC_PROJECT_2024_SORTER_CUH
#define HPC_PROJECT_2024_SORTER_CUH

#include "cub/device/device_radix_sort.cuh"

template<class KeyType, class ValueType>
class KVSorter {

public:
    explicit KVSorter(size_t len)
    : m_length(len), use_buffer(false) {
        cudaMalloc(&m_values[0], len * sizeof(ValueType));
        init();
    }
    KVSorter(size_t len, ValueType * values_buffer)
    : m_length(len), use_buffer(true) {
        m_values[0] = values_buffer;
        init();
    }
    ~KVSorter() {
        cudaFree(temp_storage);
        cudaFree(m_keys[0]);
        cudaFree(m_keys[1]);
        if(!use_buffer) cudaFree(m_values[0]);
        else if(current) cudaMemcpy(m_values[0], m_values[1], m_length * sizeof(ValueType), cudaMemcpyDeviceToDevice);
        cudaFree(m_values[1]);
    }
    inline size_t length() const { return m_length; }
    inline KeyType * keys() const { return m_keys[current]; }
    inline ValueType * values() const { return m_values[current]; }

    void sort() {
        cub::DeviceRadixSort::SortPairs(
                temp_storage, temp_storage_bytes,
                m_keys[current], m_keys[1 ^ current],
                m_values[current], m_values[1 ^ current],
                m_length);
        current ^= 1;
    }

private:
    size_t m_length, temp_storage_bytes;
    KeyType * m_keys[2];
    ValueType * m_values[2];
    void * temp_storage;
    char current = 0;
    bool use_buffer = false;

    void init() {
        cudaMalloc(&m_keys[0], m_length * sizeof(KeyType));
        cudaMalloc(&m_keys[1], m_length * sizeof(KeyType));
        cudaMalloc(&m_values[1], m_length * sizeof(ValueType));
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, m_keys[0], m_keys[1], m_values[0], m_values[1], m_length);
        cudaMalloc(&temp_storage, temp_storage_bytes);
    }
};

#endif //HPC_PROJECT_2024_SORTER_CUH
