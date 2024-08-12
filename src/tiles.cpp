#include "tiles.hpp"
#include "iostream"

Tiles::~Tiles() {
    delete[] points;
    delete[] counts;
}

void Tiles::cover(unsigned int width, unsigned int height) {
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
    delete[] counts;
    auto N = total();
    counts = new uint_fast16_t [N];
    for(uint_fast16_t i=0; i<N; i++) counts[i] = 0;
}

void Tiles::distribute(complex_t &min, complex_t &max, complex_t *particles, uint64_t N) {
    if(counts == nullptr) throw std::logic_error("Cannot distribute particles before computing tiles count");
    delete[] points;
    auto lambda = (float) N / (float) total();
    max_count = static_cast<uint32_t>(std::ceil(lambda + std::sqrt(lambda)));
    points = new complex_t [max_count*rows*cols];

    auto hscale = cols / (max.real() - min.real());
    auto vscale = rows / (max.imag() - min.imag());

    bool exceeded = false;
    for(uint64_t i=0; i<N; i++) {
        auto c = static_cast<uint_fast16_t>(hscale * (particles[i].real() - min.real()));
        auto r = static_cast<uint_fast16_t>(vscale * (particles[i].imag() - min.imag()));
        auto j = c + r*cols;
        auto offset = counts[j]++;
        if(offset < max_count) points[max_count*j + offset] = particles[i];
        else exceeded = true;
    }

    #ifdef DEBUG
    auto T = total();
    uint_fast16_t max_c = 0;
    for(uint_fast16_t i=0; i<T; i++) if(counts[i] > max_c) max_c = counts[i];
    std::cout << "[INFO] Particles memory {min: " << N << ", estimated extra: " << (max_count*T - N)
              << ", needed extra: " << max_c*T - N << ", exceeded: " << exceeded << '}' << std::endl;
    #endif
}