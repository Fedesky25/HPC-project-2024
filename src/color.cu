//
// Created by feder on 19/09/2024.
//

#include "color.cuh"

#define MANUAL_DIV_3 0
#define Over255 3.92156862745098033773416545955114997923374176025390625e-3f

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
    uint8_t bgra[4];
    bgra[3] = reduce_to_255(alpha);
    if(s == 0) {
        bgra[0] = bgra[1] = bgra[2] = reduce_to_255(l);
    }
    else {
        // S*(L<0.5? L : 1-L)  =>  s * (l<48960 ? l : 97920 - l) / 97920
        // but the product may exceed the value represented by int32_t i.e. 2^31 - 1
        // e.g. s=97920, l=48960  =>  4_794_163_200 > 2_147_483_648 = 2^31 - 1
        // therefore pre-divide both side by 2 and finally divide by 97920/4 = 24480
        auto delta = (s >> 1) * ((l < icenc(0.5) ? l : ICE_1-l) >> 1) / icenc(0.25);
        bgra[0] = fixed_ldt_to_component(l, delta, h - icenc_inv(3.0));    // blue
        bgra[1] = fixed_ldt_to_component(l, delta, h);                      // green
        bgra[2] = fixed_ldt_to_component(l, delta, h + icenc_inv(3.0));    // red
    }
    uint32_t result = *((uint32_t*)bgra);
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
