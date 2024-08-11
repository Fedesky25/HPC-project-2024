#include "utils.cuh"
#include <cstdlib>
#include <iostream>


inline unsigned udist(unsigned a, unsigned b) {
    return (a > b) ? a-b : b-a;
}

void TilesCount::cover(unsigned int width, unsigned int height) {
    unsigned rev = 0;
    if(height > width) {
        rev = width;
        width = height;
        height = rev;
    }
    float min = INFINITY;
    float ratio = (float) width / (float) height;
    for(unsigned r=1; r <= 32; r++) {
        auto c = static_cast<unsigned>(std::round(ratio*r));
        while(r * c > 1024) c--;
        auto d = std::abs((float) c / (float) r - ratio);
        if(d <= min) {
            rows = r;
            cols = c;
            min = d;
        }
    }
    if(rev) {
        rev = rows;
        rows = cols;
        cols = rev;
    }
}

PixelIndex Canvas::where(complex_t z) const {
    auto row = static_cast<int32_t>(std::round(z.real() - center.real())) + (width >> 1);
    auto col = static_cast<int32_t>(std::round(z.imag() - center.imag())) + (height >> 1);
    if(row < 0 || row >= width || col < 0 || col > height) return {};
    else return { row, col };
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

unsigned long Configuration::particle_number() const {
    auto extra = 2*margin*particle_distance;
    return (canvas.width + extra) * (canvas.height + extra) / (particle_distance*particle_distance);
}

TilesCount::TilesCount(Configuration &config) {
    auto extra = config.margin*config.particle_distance*2;
    cover(config.canvas.width+extra, config.canvas.height+extra);
}
