// The possible hue of particle span the range [0, 2/3] and we encode it in an uint16_t
// The integer value M representing 2/3 is chosen such that
//  - 3/2*M, which encodes 1, should be an exact int
//  - 3/4*M, which encodes 0.5, should be an exact int
//  - the down-conversion to a 8-bit range [0,255] is easy
// Thus we choose M = 255 * 256 = 65280  =>  3/2*M = 97920
// The down-conversion is easy since x -> 255/97920*x = x / (3*128) = (x>>7)/3

#ifndef HPC_PROJECT_2024_COLOR_CUH
#define HPC_PROJECT_2024_COLOR_CUH

#include "cstdint"

#define ICE_1 97920

/**
 * Integer Color Encode
 * @param x number in range [0,1]
 * @return integer representation of x
 */
constexpr inline int32_t icenc(float x) { return static_cast<int32_t>(ICE_1 * x); }

/**
 * Integer Color Encode
 * @param x number in range (0,+infinity)
 * @return integer representation of 1/x
 */
constexpr inline int32_t icenc_inv(float x) { return static_cast<int32_t>(ICE_1 / x); }

struct FixedFraction {
    int32_t value;

    /**
     * @param a numerator, must be less than denominator
     * @param b denominator
     */
    inline FixedFraction(int32_t a, int32_t b) : value((a<<14) / b) {}

    /**
     * Multiplies this fraction with a number
     * @param x other number
     * @return
     */
    inline int32_t multiply(int32_t x) const { return (x * value) >> 14; }

    inline int32_t mix(int32_t start, int32_t end) const {
        return start + (((end - start)*value) >> 14);
    }
};

/**
 * Transform an HSLA value using 17-bit encoding in range [0, 98302] to the 8-bit encoded RGBA value
 * @see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
 * @param h hue
 * @param s saturation
 * @param l lightness
 * @return RGBA value as a uint32_t
 */
__device__ __host__ uint32_t HSLA_to_RGBA(int32_t h, int32_t s, int32_t l, int32_t alpha);

/** Representation of HSLA color using integers */
struct FixedHSLA {
    int32_t H, S, L, A;

    /**
     * Computes the integer-encoded HSLA from float-encoded HSLA
     * @param hue
     * @param saturation
     * @param lightness
     * @param alpha
     */
    void fromFloatHSLA(float hue, float saturation, float lightness, float alpha);

    /**
     * Converts the 4-byte RGBA to HSLA
     * @return RGBA 4-byte format as a single uint32
     */
    void fromRGBA(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha);

    /**
     * Converts the current HSLA color to RGBA format
     * @return RGBA 4-byte format as a single uint32
     */
    __device__ __host__ inline uint32_t toRGBA() const { return HSLA_to_RGBA(H, S, L, A); }

    /**
     * Mix this color with another one
     * @param other other HSLA color
     * @param x how much the other color is present
     */
    __device__ __host__ void mixWith(const FixedHSLA & other, FixedFraction frac);
};

#endif //HPC_PROJECT_2024_COLOR_CUH
