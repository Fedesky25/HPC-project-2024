//
// Created by feder on 19/09/2024.
//

#include "color.cuh"
#include <cuda/std/cmath>

#define MANUAL_DIV_3 0
#define Over255 3.92156862745098033773416545955114997923374176025390625e-3f

#if __cplusplus >= 201700L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201700L)
    #define SAFE_IF_CONSTEXPR constexpr
#else
    // we hope compiler removes unused branch
    #define SAFE_IF_CONSTEXPR
    #if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
        #warning C++ version is 14 or less
    #elif defined(_MSC_VER) || defined(__clang__)
        #pragma message("C++ version is 14 or less (" __c)
    #endif
#endif

/**
 * Perform the rounded division by 3*128
 * @param x integer value between 0 and 97920
 * @return
 */
__device__ __host__ inline uint8_t reduce_to_255(int32_t x) {
    #if MANUAL_DIV_3
    // x/(128*3) --> (2x/(3*128) + 1)/2 = (x/(3*64) + 1)>>1;
    // since 1/3 = 1/4 + 1/16 + 1/64 + 1/256 + 1/1024 + ...
    // divide by 3 up with powers up to 4^6
    // because 97920 / 4^6 = 97920 >> 12 = 23 < 2^6 = 64
    // and     97920 / 4^5 = 97920 >> 10 = 95 > 2^6 = 64
    int32_t q, y= 0;
    x = x >> 6;
    for(int i=0; i<6; i++) {
        q = x >> 2;
        x = q + (x&3);
        y += q;
    }
    return (y+1) >> 1;
    #else
    return (x/192 + 1) >> 1;
    #endif
}

__device__ __host__ uint8_t fixed_ldt_to_component(int32_t l, int32_t d, int32_t t) {
    if(t < 0) t += ICE_1;
    else if(t > ICE_1) t -= ICE_1;
    int32_t result = l - d;
    if(t < icenc_inv(6.0)) result += (d*t)/icenc_inv(12.0);
    else if(t < icenc(0.5)) result += 2*d;
    else if(t < icenc(2.0/3.0)) result += d*(icenc(2.0/3.0) - t)/icenc_inv(12.0);
    return reduce_to_255(result);
}


__device__ __host__ uint32_t HSLA_to_RGBA(int32_t h, int32_t s, int32_t l, int32_t alpha) {
    uint8_t rgba[4];
    rgba[3] = reduce_to_255(alpha);
    if(s == 0) {
        rgba[0] = rgba[1] = rgba[2] = reduce_to_255(l);
    }
    else {
        // S*(L<0.5? L : 1-L)  =>  s * (l<48960 ? l : 97920 - l) / 97920
        // but the product may exceed the value represented by int32_t i.e. 2^31 - 1
        // e.g. s=97920, l=48960  =>  4_794_163_200 > 2_147_483_648 = 2^31 - 1
        // therefore pre-divide both side by 2 and finally divide by 97920/4 = 24480
        auto delta = (s >> 1) * ((l < icenc(0.5) ? l : ICE_1-l) >> 1) / icenc(0.25);
        rgba[0] = fixed_ldt_to_component(l, delta, h + icenc_inv(3.0));    // red
        rgba[1] = fixed_ldt_to_component(l, delta, h);                      // green
        rgba[2] = fixed_ldt_to_component(l, delta, h - icenc_inv(3.0));    // blue
    }
    uint32_t result = *((uint32_t*)rgba);
    return result;
}

void FixedHSLA::fromFloatHSLA(float hue, float saturation, float lightness, float alpha) {
    H = icenc(hue);
    S = icenc(saturation);
    L = icenc(lightness);
    A = icenc(alpha);
}

__device__ __host__ void FixedHSLA::mixWith(const FixedHSLA &other, FixedFraction frac) {
    H = frac.mix(H, other.H);
    S = frac.mix(S, other.S);
    L = frac.mix(L, other.L);
    A = frac.mix(A, other.A);
}

void FixedHSLA::fromRGBA(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) {
    A = ((int32_t)alpha << 7) * 3;

    float R = (float)red/255.0f, G = (float)green/255.0f, B = (float)blue/255.0f;
    float vmax = std::max(R, std::max(G, B));
    float vmin = std::min(R, std::min(G, B));

    auto sum = vmax + vmin;
    L = static_cast<int32_t>(sum * icenc(0.5));

    if(vmax == vmin) {
        H = 0;
        S = 0;
    }
    else {
        if(sum > 1.0) sum = 2.0f - sum;
        auto d = vmax - vmin;
        S = static_cast<int32_t>(ICE_1 * d / sum);
        float h;
        if (vmax == R) h = (G - B) / d + (G < B ? 6.0f : 0.0f);
        else if (vmax == G) h = (B - R) / d + 2;
        else h = (R - G) / d + 4;
        H = static_cast<int32_t>(ICE_1 * h / 6.0f);
    }
}


#define SATURATION 0.55f
#define LIGHTNESS 0.55f

const constexpr auto delta_hue = 2 * SATURATION * (1.0f - LIGHTNESS);
const constexpr auto color_normalizer = 1.0f / ICE_1;

__device__ __host__ float component_from_t(float t) {
    // @see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
    float result = LIGHTNESS - 0.5f*delta_hue;
    if(t < 1.0f/6.0f) result += delta_hue*6*t;
    else if(t < 0.5f) result += delta_hue;
    else if(t < 2.0f/3.0f) result += delta_hue*6*(2.0f/3.0f - t);
    return result;
}

__device__ __host__ void RGBA::from_hue(uint16_t hue) {
    float H = (float) hue * color_normalizer;
    G = component_from_t(H);
    float t = H + 1.0f/3.0f;
    if(t > 1.0f) t -= 1.0f;
    R = component_from_t(t);
    t = H - 1.0f/3.0f;
    if(t < 0.0f) t += 1.0f;
    B = component_from_t(t);
}

template<bool opaque>
__device__ __host__ void RGBA::over(const RGBA * backdrop) {
    float cA = 1.0f - A;
    if SAFE_IF_CONSTEXPR (!opaque) cA *= backdrop->A;
    R = R*A + backdrop->R*cA;
    G = G*A + backdrop->G*cA;
    B = B*A + backdrop->B*cA;
    if SAFE_IF_CONSTEXPR (opaque) A = 1.0f;
    else {
        A += cA;
        float f = 1.0f/A;
        R *= f;
        G *= f;
        B *= f;
    }
}

template __device__ __host__ void RGBA::over<false>(const RGBA *backdrop);
template __device__ __host__ void RGBA::over<true>(const RGBA *backdrop);

template<bool opaque>
__device__ __host__ void RGBA::write(unsigned char * buffer) const {
    buffer[0] = static_cast<unsigned char>(cuda::std::round(255*R));
    buffer[1] = static_cast<unsigned char>(cuda::std::round(255*G));
    buffer[2] = static_cast<unsigned char>(cuda::std::round(255*B));
    if SAFE_IF_CONSTEXPR (!opaque) buffer[3] = static_cast<unsigned char>(cuda::std::round(255*A));
}

template __device__ __host__ void RGBA::write<false>(unsigned char *buffer) const;
template __device__ __host__ void RGBA::write<true>(unsigned char *buffer) const;
