#include "cli.hpp"
#include <iostream>
#include "getopt.h"

#define INVALID_OPTION(NAME) { std::cerr << '"' << NAME << "\" is not a valid option" << std::endl; return true; }
#define INVALID_RESOLUTION_NAME std::cerr << "Invalid screen resolution name" << std::endl;
#define CHECK_MISSING(NAME) if(i+1 == argc) {               \
    std::cerr << "Missing value for " << NAME << std::endl; \
    return true;                                           \
}
#define CHECK_REMAINING(NAME) if(*rest != '\0') {                           \
    std::cerr << "Malformed format of the value of " << NAME << std::endl;  \
    return true;                                                           \
}
#define CHECK_DISTANCE if(config->particle_distance < 1) {                       \
    std::cerr << "Particle distance cannot be lower than 1" << std::endl;\
    return true;                                                       \
}
#define CHECK_COMPLEX(VAR, NAME) if(isnan(VAR)) { \
    std::cerr << "Malformed complex number format (" << NAME << ')' << std::endl; \
    return true;                                  \
}
#define INVALID_SCALE_UNIT { std::cerr << "Invalid unit of scale" << std::endl; return 0.0; }





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

inline bool isnan(complex_t z) { return isnan(z.real()) || isnan(z.imag()); }

void parse_resolution(const char * str, CanvasAdapter * canvas) {
    unsigned width = 0, height = 0;
    bool rotate = false;
    if(*str == '^') {
        rotate = true;
        str += 1;
    }
    if(isdigit(*str)) {
        int consumed;
        int c = sscanf_s(str, "%ux%u%n", &width, &height, &consumed);
        if(c != 2 || str[consumed] != '\0') {
            std::cerr << "Custom resolution format invalid" << std::endl;
            width = height = 0;
        }
        else if(width > 8192 || height > 8192) {
            std::cerr << "Resolution must be at most 8192x8192" << std::endl;
            width = height = 0;
        }
    }
    else {
        unsigned len = strlen(str);
        switch (len) {
            case 1:
            case 2:
                INVALID_RESOLUTION_NAME
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
                        INVALID_RESOLUTION_NAME
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
                        INVALID_RESOLUTION_NAME
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
                        INVALID_RESOLUTION_NAME
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
                        INVALID_RESOLUTION_NAME
                }
                break;
            default:
                INVALID_RESOLUTION_NAME
        }
    }
    if(rotate) {
        unsigned t = width;
        width = height;
        height = t;
    }
    canvas->width = width;
    canvas->height = height;
}

enum class ScaleScaling { NONE, WIDTH, HEIGHT };

double parse_scale(const char * str, ScaleScaling * action) {
    char * rest;
    double v = strtod(str, &rest);
    if(v <= 0.0) {
        std::cerr << "Scale must be positive" << std::endl;
        return 0.0;
    }
    str = rest;
    switch (*str) {
        case 'h':
        case 'w':
            if(str[1] != '/' || str[2] != 'u' || str[3] != '\0') INVALID_SCALE_UNIT
            *action = str[0] == 'h' ? ScaleScaling::HEIGHT : ScaleScaling::WIDTH;
            break;
        case 'p':
            if(str[1] != 'x' || str[2] != '/' || str[3] != 'u' || str[4] != '\0') INVALID_SCALE_UNIT
            *action = ScaleScaling::NONE;
            break;
        case 'u':
            if(str[1] != '/') INVALID_SCALE_UNIT
            v = 1/v;
            switch (str[2]) {
                case 'h':
                    if(str[3] != '\0') INVALID_SCALE_UNIT
                    *action = ScaleScaling::HEIGHT;
                    break;
                case 'w':
                    if(str[3] != '\0') INVALID_SCALE_UNIT
                    *action = ScaleScaling::WIDTH;
                    break;
                case 'p':
                    if(str[3] != 'x' || str[4] != '\0') INVALID_SCALE_UNIT
                    *action = ScaleScaling::NONE;
                    break;
                default:
                    INVALID_SCALE_UNIT
            }
            break;
        default:
            INVALID_SCALE_UNIT
    }
    return v;
}

bool parse_args(int argc, char * argv[], Configuration * config) {
    static option long_options[] = {
            { "parallel",    required_argument, nullptr, 'p' },
            { "output",      required_argument, nullptr, 'o' },
            { "resolution",  required_argument, nullptr, 'R' },
            { "pixel-scale", required_argument, nullptr, 's' },
            { "center",      required_argument, nullptr, 'c' },
            { "distance",    required_argument, nullptr, 'd' },
            { "margin",      required_argument, nullptr, 'm' },
            { "lloyd",       required_argument, nullptr, 'L' },
            { "speed",       required_argument, nullptr, 'v' },
            { "time-scale",  required_argument, nullptr, 't' },
            { "framerate",   required_argument, nullptr, 'f' },
            { "duration",    required_argument, nullptr, 'D' },
            { "lifetime",    required_argument, nullptr, 'l' },
            { "background",  required_argument, nullptr, 'B' },
            { "int",         required_argument, nullptr, 'n' },
            { "real",        required_argument, nullptr, 'r' },
            { "complex1",    required_argument, nullptr, '1' },
            { "complex2",    required_argument, nullptr, '2' },
            { "complex3",    required_argument, nullptr, '3' },
            { nullptr, 0, nullptr, 0 }
    };
    static char short_options[] = "p:o:R:s:c:d:m:L:v:t:f:D:l:B:n:r:1:2:3:";

    int o, go = 1, index_opt;
    char * rest;
    double time_scale = 0.12;
    unsigned long fps = 60, duration = 10, lifetime = 8;
    ScaleScaling action = ScaleScaling::NONE;

    while(go) {
        o = getopt_long(argc, argv, short_options, long_options, &index_opt);
        switch (o) {
            case -1:
            case '?':
                go = false;
                break;
            case 'p':
                if(strcmp(optarg, "none") == 0) config->mode = ExecutionMode::Serial;
                else if(strcmp(optarg, "omp") == 0) config->mode = ExecutionMode::OpenMP;
                else if(strcmp(optarg, "gpu") == 0) config->mode = ExecutionMode::GPU;
                else {
                    std::cerr << "Parallelization option must be one of: none, omp, gpu" << std::endl;
                    return true;
                }
                break;
            case 'o':
                config->output = optarg;
                break;
            case 'm':
                config->margin = strtoul(optarg, &rest, 10);
                break;
            case 'c':
                config->canvas.center = parse_complex(optarg);
                CHECK_COMPLEX(config->canvas.center, "center")
                break;
            case 's':
                config->canvas.scale = parse_scale(optarg, &action);
                if(config->canvas.scale == 0.0) return true;
                break;
            case 'd':
                config->particle_distance = strtoul(optarg, &rest, 10);
                CHECK_DISTANCE
                break;
            case 'L':
                config->lloyd_iterations = strtoul(optarg, &rest, 10);
                CHECK_REMAINING("number of iterations of Lloyd's algorithm")
                break;
            case 'R':
                parse_resolution(optarg, &(config->canvas));
                if(!config->canvas.width) return true;
                break;
            case 'v':
            {
                double v = strtod(optarg, &rest);
                CHECK_REMAINING("speed (v)")
                config->evolution.speed_factor = v*v;
                break;
            }
            case 'f':
                fps = strtoul(optarg, &rest, 10);
                CHECK_REMAINING("fps")
                break;
            case 't':
                time_scale = strtod(optarg, &rest);
                CHECK_REMAINING("time-scale")
                if(time_scale <= 0) {
                    std::cerr << "The time scale must be strictly positive" << std::endl;
                    return true;
                }
                break;
            case 'D':
                duration = strtoul(optarg, &rest, 10);
                CHECK_REMAINING("duration")
                break;
            case 'l':
                lifetime = strtoul(optarg, &rest, 10);
                CHECK_REMAINING("lifetime")
                break;
            case 'B':
            {
                uint32_t val = strtoul(optarg, &rest, 16);
                CHECK_REMAINING("background color")
                config->background.A = 1.0f;
                if(rest - optarg > 6) {
                    config->background.A = (float)(val & 0xff) * div_255;
                    val >>= 8;
                }
                config->background.R = (float)(val>>16) * div_255;
                config->background.G = (float)((val>>8)&0xff) * div_255;
                config->background.B = (float)(val&0xff) * div_255;
                break;
            }
            case 'n':
                config->vars.n = strtol(optarg, &rest, 10);
                CHECK_REMAINING("integer number (n)");
                break;
            case 'r':
                config->vars.x = strtod(optarg, &rest);
                CHECK_REMAINING("real number (x)");
                break;
            case '1':
            case '2':
            case '3':
            {
                auto index = o - '1';
                config->vars.z[index] = parse_complex(optarg);
                CHECK_COMPLEX(config->vars.z[index], 'z' << index+1)
                break;
            }
        }
    }
    switch (action) {
        case ScaleScaling::WIDTH:
            config->canvas.scale *= config->canvas.width;
            break;
        case ScaleScaling::HEIGHT:
            config->canvas.scale *= config->canvas.height;
            break;
        case ScaleScaling::NONE:
            break;
    }
    if(lifetime > duration) {
        std::cerr << "Particles' lifetime cannot exceed the video duration" << std::endl;
        return true;
    }
    config->evolution.delta_time = time_scale / fps;
    config->evolution.frame_count = duration * fps;
    config->evolution.frame_rate = fps;
    config->evolution.life_time = lifetime * fps;
    if(config->evolution.frame_count > 64800) {
        std::cerr << "Number of frames cannot exceed 64800 i.e. 4:30min at 240Hz or 18min at 60Hz" << std::endl;
        return true;
    }
    return false;
}

void print_usage() {
    std::cout << "CFSPlt v0.1.0" << std::endl;
    std::cout << "  Complex functions streamplot  -  Generator of webp videos representing the streamplot of a selection of complex functions" << std::endl << std::endl;
    std::cout << "SYNOPSIS:" << std::endl;
    std::cout << "  cfsplt [-c center_point] [-d particle_distance] [-D duration] [-f fps] [-m margin_layers] [-n integer] [-o file] [-p parallelization]" << std::endl
              << "         [-r real] [-R resolution] [-s pixel_scale] [-t time_scale] [-v speed] [-1 complex] [-2 complex] [-3 complex] function" << std::endl << std::endl;
    std::cout << "OPTIONS" << std::endl;
    std::cout << "  Name                 Default        Description" << std::endl;
    std::cout << "  -p  --parallel       gpu            Which parallelization to adopt in computations. It must be one of: none, omp, gpu" << std::endl;
    std::cout << "  -o  --output         plot.raw       Path of the output raw file" << std::endl;
    std::cout << "  -D  --duration       10             Duration in seconds of the webp animation" << std::endl;
    std::cout << "  -f  --framerate      60             Number of frames per seconds i.e. the refresh rate" << std::endl;
    std::cout << "  -R  --resolution     1920x1080      Pixel sizes of the video: it can be either a supported screen resolution name (such as FHD, WXGA+)" << std::endl
              << "                                      or a custom size specified in the format <width>x<height>. Optionally, the character '^' may be" << std::endl
              << "                                      prepended to invert the horizontal and vertical sizes." << std::endl;
    std::cout << "  -B  --background     242429         RGB or RGBA color of the background using hexadecimal representation" << std::endl;
    std::cout << "  -d  --distance       10             Average distance (in pixels) between two nearby particles in the starting positions" << std::endl;
    std::cout << "  -m  --margin         4              Number of layers of additional particles outside the video. Too low values lead to empty borders." << std::endl;
    std::cout << "  -L  --lloyd          8              Particles' lifetime in seconds; must be less than the video duration." << std::endl;
    std::cout << "  -l  --lifetime       7              Number of iterations of Lloyd's algorithm to evenly distribute particles." << std::endl;
    std::cout << "  -v  --speed          1.0            Value of speed around which logarithmic color sensitivity is maximum. Red or blue" << std::endl
              << "                                      occur when the speed is respectively one order less or more than the specified value." << std::endl;
    std::cout << "  -t  --time-scale     0.12           Time scale used to convert 1 real second into the computational time unit. Lower values guarantee" << std::endl
              << "                                      a more precise computation of the particle evolution at the cost of less motion." << std::endl;
    std::cout << "  -s  --pixel-scale    100px/u        Scale used to convert distance between complex numbers to pixels. The required unit must be one of:" << std::endl
              << "                                      u/px, u/w, u/h, px/u, w/u, h/u; where 'px' is pixel, 'w' is the width of the video (in pixel), 'h'" << std::endl
              << "                                      is the height of the video (in pixel), and 'u' is the unitary distance" << std::endl;
    std::cout << "  -c  --center         0+0i           The complex number at the center of the video. See later on supported complex number formats." << std::endl;
    std::cout << "  -n  --int            0              Integer number used in some functions" << std::endl;
    std::cout << "  -r  --real           3.14159...     Real number used in some functions" << std::endl;
    std::cout << "  -1  --complex1       1              First complex number used in some functions" << std::endl;
    std::cout << "  -2  --complex2       1i             Second complex number used in some functions" << std::endl;
    std::cout << "  -3  --complex3       1\\45d          Third complex number used in some functions" << std::endl << std::endl;
    std::cout << "COMPLEX NUMBER FORMAT" << std::endl;
    std::cout << "  Complex number can be specified in cartesian and polar coordinates:" << std::endl;
    std::cout << "  - Cartesian format is the sum of a real and imaginary part. The latter is denote by prepending or appending 'i' or 'j' to the number" << std::endl;
    std::cout << "  - Polar coords. are in the format <radius>\\<angle><unit>, where the angle unit can be degree (d), radian (r), gon (g), or turns (t)" << std::endl;
    std::cout << std::endl;
}