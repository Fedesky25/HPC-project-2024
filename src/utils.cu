#include "utils.cuh"
#include <cstdlib>
#include <iostream>


__device__ __host__ int32_t CanvasAdapter::where(complex_t z) {
    auto col = static_cast<int32_t>(std::round(scale*(z.real() - center.real()))) + (int32_t)(width >> 1);
    auto row = static_cast<int32_t>(std::round(scale*(z.imag() - center.imag()))) + (int32_t)(height >> 1);
    if(row < 0 || row >= height || col < 0 || col > width) return -1;
    else return col + row * (int32_t)width;
}

std::ostream &operator<<(std::ostream &os, CanvasAdapter &cv) {
    return os << cv.width << 'x' << cv.height << '@' << cv.center << '$' << cv.scale << "px/u";
}

std::ostream &operator<<(std::ostream &os, EvolutionOptions &eo) {
    return os << eo.frame_count << "f, " << eo.frame_rate << "f/s, dt=" << eo.delta_time << ", @v=" << eo.speed_factor;
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