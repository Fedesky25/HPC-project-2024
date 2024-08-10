#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <complex>
#include <cstdint>

template<unsigned N>
constexpr uint64_t str_to_num(const char str[N+1]) {
    static_assert(N < 9, "Number of characters cannot exceed 8");
    auto res = (uint64_t)str[0];
    for(unsigned i=1; i<N; i++) res += (uint64_t)str[i] << i*8;
    return res;
}

using complex_t = std::complex<double>;

complex_t parse_complex(const char * str);

struct ScreenResolution {
    unsigned width = 1920, height = 1080;
    ScreenResolution() = default;
    inline explicit ScreenResolution(const char * str) { parse(str); }
    inline explicit operator bool () const { return width && height; }
    void parse(const char * str);
private:
    inline void set_invalid() { width = height = 0; }
};

struct TilesCount {
    uint16_t rows = 0, cols = 0;
    inline TilesCount() = default;
    inline TilesCount(unsigned width, unsigned height) { cover(width, height); }
    inline explicit TilesCount(ScreenResolution res) { cover(res.width, res.height); }
    void cover(unsigned width, unsigned height);
    inline uint16_t total() const { return rows * cols; }
};

#endif //HPC_PROJECT_2024_UTILS_CUH
