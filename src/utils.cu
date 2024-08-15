#include "utils.cuh"
#include <cstdlib>
#include <iostream>


inline unsigned udist(unsigned a, unsigned b) {
    return (a > b) ? a-b : b-a;
}

PixelIndex Canvas::where(complex_t z) const {
    auto row = static_cast<int32_t>(std::round(z.real() - center.real())) + (int32_t)(width >> 1);
    auto col = static_cast<int32_t>(std::round(z.imag() - center.imag())) + (int32_t)(height >> 1);
    if(row >= height) row = -1;
    if(col >= width) col = -1;
    return { row, col };
}

std::ostream &operator<<(std::ostream &os, Canvas &cv) {
    return os << cv.width << 'x' << cv.height << '@' << cv.center << '$' << cv.scale << "px/u";
}

void Configuration::bounds(complex_t *min, complex_t *max) const {
    auto extra = 2*margin*particle_distance;
    double dr = (canvas.width + extra) / canvas.scale * 0.5;
    double di = (canvas.height + extra) / canvas.scale * 0.5;
    min->real(canvas.center.real() - dr);
    min->imag(canvas.center.imag() - di);
    max->real(canvas.center.real() + dr);
    max->imag(canvas.center.imag() + di);
}

void Configuration::sizes(unsigned int *width, unsigned int *height) const {
    auto extra = 2*margin*particle_distance;
    *width = canvas.width + extra;
    *height = canvas.height + extra;
}

uint32_t Configuration::particle_number() const {
    auto extra = 2*margin*particle_distance;
    return (canvas.width + extra) * (canvas.height + extra) / (particle_distance*particle_distance);
}

template<class T, class I>
__device__ __host__ I lower_bound(T value, T * data, I length)
{
    static_assert(std::is_integral_v<I>, "Type used as index must be of integral type");
    I step, index, first = 0;
    while (length > 0) {
        index = first;
        step = length >> 1;
        index += step;
        if (data[index] < value) {
            first = index + 1;
            length -= step + 1;
        }
        else length = step;
    }
    return first;
}
