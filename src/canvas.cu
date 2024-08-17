#include "canvas.cuh"

#define SATURATION 0.55
#define LIGHTNESS 0.55

#define BG_R 21
#define BG_G 21
#define BG_B 25

__device__ __host__ bool CanvasPixel::update_age(uint16_t _age) {
    bool ok = true;
    if(age == UINT16_MAX) age = _age;
    else if(age + multiplicity + 1 == _age) multiplicity++;
    else ok = false;
    return ok;
}

/**
 * @see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
 * @param t
 * @return
 */
__device__ __host__ double xToRGB(double t) {
    constexpr double q = LIGHTNESS + SATURATION - LIGHTNESS*SATURATION;
    constexpr double p = 2.0*LIGHTNESS - q;
    constexpr double f = (q-p)*2.0;

    if (t < 0.0) t += 3.0;
    if (t > 3.0) t -= 3.0;
    if (t < 0.5) return p + f * t;
    if (t < 1.5) return q;
    if (t < 2.0) return p + f * (2 - t);
    return p;
}

__device__ __host__ void CanvasPixel::set_color(double square_speed, double factor) {
    // saturation & lightness are fixed to 0.55
    auto x = 2.0 * square_speed / (square_speed + factor); // 1.0 / (1.0 + factor/ square_speed)
    // h = x / 3
    color.r = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x + 1)));
    color.g = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x)));
    color.b = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x - 1)));
}

__device__ __host__ ARGB CanvasPixel::get_color(uint16_t time) const {
    ARGB result;
    if(time < age || time >= age + multiplicity + 200) {
        result.r = BG_R;
        result.g = BG_G;
        result.b = BG_B;
    }
    else if(time <= age + multiplicity) result = color;
    else {
         auto x = (time - age - multiplicity) * 0.005;
         result.r = static_cast<uint8_t>(cuda::std::round(x*BG_R + (1-x)*color.r));
         result.g = static_cast<uint8_t>(cuda::std::round(x*BG_G + (1-x)*color.g));
         result.b = static_cast<uint8_t>(cuda::std::round(x*BG_B + (1-x)*color.b));
    }
    return result;
}

