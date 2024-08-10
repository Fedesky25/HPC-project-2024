#include "utils.cuh"
#include <cstdlib>
#include <iostream>

#define PI 3.1415926535897932384626433

double parse_complex_last_number(const char * str, char** rest) {
    switch (*str) {
        case '\0':
            return 0.0;
        case '-':
        case '+':
        {
            double v = strtod(str, rest);
            return *rest == str || **rest != 0 ? NAN : v;
        }
        default:
            return NAN;
    }
}

complex_t parse_complex(const char * str) {
    char * rest;
    if(str[0] == 'i' || str[0] == 'j') {
        double v = strtod(str + 1, &rest);
        str = rest;
        return {parse_complex_last_number(str, &rest), v };
    }
    else {
        double r = strtod(str, &rest);
        if(rest == str) return { NAN, NAN };
        str = rest;
        switch (*str) {
            case '\0':
                return {r, 0.0};
            case 'i':
            case 'j':
                return {parse_complex_last_number(str + 1, &rest), r };
            case '+':
            case '-':
                switch (str[1]) {
                    case '\0':
                        return {r, NAN};
                    case 'i':
                    case 'j':
                    {
                        if(isdigit(str[2])){
                            double i = strtod(str+2, &rest);
                            return {r, str[0] == '-' ? -i : i};
                        }
                        else {
                            return {r, NAN};
                        }
                    }
                    case '0':
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                    {
                        double i = strtod(str, &rest);
                        str = rest;
                        if((*str == 'i' || *str =='j') && str[1] == '\0') return {r,i};
                        else return {r,NAN};
                    }
                    default:
                        return {r,NAN};
                }
            case '\\':
            {
                double angle = strtod(str+1, &rest);
                bool check_end = true;
                str = rest;
                switch (*str) {
                    case '\0':
                        check_end = false;
                    case 'r':
                        break;
                    case 'd':
                        angle *= PI / 180.0;
                        break;
                    case 'g':
                        angle *= PI * 0.005;
                        break;
                    case 't':
                        angle *= 2*PI;
                        break;
                    default:
                        return {NAN, NAN};
                }

                if(!check_end || str[1] == '\0') return {r*std::cos(angle), r*std::sin(angle)};
                else return {NAN, NAN};
            }
            default:
                return {NAN,NAN};
        }
    }
}

void ScreenResolution::parse(const char *str) {
    bool rotate = false;
    if(*str == '^') {
        rotate = true;
        str += 1;
    }
    if(isdigit(*str)) {
        int consumed;
        int c = sscanf_s(str, "%ux%u%n", &width, &height, &consumed);
        if(c != 2 || str[consumed] != '\0') set_invalid();
    }
    else {
        unsigned len = strlen(str);
        switch (len) {
            case 1:
            case 2:
                set_invalid();
                break;
            case 3:
                switch (str_to_num<3>(str)) {
                    case str_to_num<3>("VGA"):
                        width = 640;
                        height = 480;
                        break;
                    case str_to_num<3>("XGA"):
                        width = 1024;
                        height = 768;
                        break;
                    case str_to_num<3>("qHD"):
                        width = 960;
                        height = 540;
                        break;
                    case str_to_num<3>("FHD"):
                        width = 1920;
                        height = 1080;
                        break;
                    case str_to_num<3>("UHD"):
                        width = 3840;
                        height = 2160;
                        break;
                    default:
                        set_invalid();
                }
                break;
            case 4:
                switch (str_to_num<4>(str)) {
                    case str_to_num<4>("SVGA"):
                        width = 800;
                        height = 600;
                        break;
                    case str_to_num<4>("QXGA"):
                        width = 2048;
                        height = 1536;
                        break;
                    case str_to_num<4>("WQHD"):
                        width = 2560;
                        height = 1440;
                        break;
                    case str_to_num<4>("FHD+"):
                        width = 1920;
                        height = 1280;
                        break;
                    case str_to_num<4>("WXQA"):
                        width = 1280;
                        height = 800;
                        break;
                    default:
                        set_invalid();
                }
                break;
            case 5:
                switch (str_to_num<5>(str)) {
                    case str_to_num<5>("UWFHD"):
                        width = 2560;
                        height = 1080;
                        break;
                    case str_to_num<5>("UWQHD"):
                        width = 3440;
                        height = 1440;
                        break;
                    case str_to_num<5>("WXGA+"):
                        width = 1440;
                        height = 900;
                        break;
                    case str_to_num<5>("WUXGA"):
                        width = 1920;
                        height = 1200;
                        break;
                    case str_to_num<5>("WQXGA"):
                        width = 2560;
                        height = 1600;
                        break;
                    default:
                        set_invalid();
                }
                break;
            case 6:
                switch (str_to_num<6>(str)) {
                    case str_to_num<6>("WQXGA+"):
                        width = 3200;
                        height = 1800;
                        break;
                    case str_to_num<6>("UW-FHD"):
                        width = 2560;
                        height = 1080;
                        break;
                    case str_to_num<6>("UW-QHD"):
                        width = 3440;
                        height = 1440;
                        break;
                    case str_to_num<6>("WSXGA+"):
                        width = 1680;
                        height = 1050;
                        break;
                    case str_to_num<6>("WQUXGA"):
                        width = 3840;
                        height = 2400;
                        break;
                    default:
                        set_invalid();
                }
                break;
            default:
                set_invalid();
        }
    }
    if(rotate) {
        unsigned t = width;
        width = height;
        height = t;
    }
}

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