#include "cli.cuh"
#include <iostream>

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
    std::cerr << "Paricle distance cannot be lower than 1" << std::endl;\
    return true;                                                       \
}

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

void parse_resolution(const char * str, Canvas * canvas) {
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

void print_usage() {

}

bool parse_args(int argc, char * argv[], Configuration * config) {
    char * rest;
    for(int i=1; i<argc; i++) {
        if(argv[i][0] != '-') INVALID_OPTION(argv[i])
        if(argv[i][1] == '-') {
            // long name
            const char * str = argv[i]+2;
            if(strcmp(str, "distance") == 0) {
                CHECK_MISSING("particle distance")
                config->particle_distance = strtoul(argv[++i], &rest, 10);
                CHECK_REMAINING("particle distance")
                CHECK_DISTANCE
            }
            else if(strcmp(str, "screen") == 0) {
                CHECK_MISSING("screen resolution")
                parse_resolution(argv[++i], &(config->canvas));
                if(!config->canvas.width) return true;
            }
            else INVALID_OPTION(argv[i]+2)
        }
        else {
            // short name
            switch (argv[i][1]) {
                case 'd':
                    CHECK_MISSING("particle distance (d)")
                    config->particle_distance = strtoul(argv[++i], &rest, 10);
                    CHECK_REMAINING("particle distance (d)")
                    CHECK_DISTANCE
                    break;
                case 's':
                    CHECK_MISSING("screen resolution (s)")
                    parse_resolution(argv[++i], &(config->canvas));
                    if(!config->canvas.width) return true;
                    break;
                case 'n':
                    CHECK_MISSING("integer number (n)");
                    config->vars.n = strtol(argv[++i], &rest, 10);
                    CHECK_REMAINING("integer number (n)");
                    break;
                case 'x':
                    CHECK_MISSING("real number (x)");
                    config->vars.x = strtod(argv[++i], &rest);
                    CHECK_REMAINING("real number (x)");
                    break;
                case 'z':
                {
                    auto index = argv[i][2] - '1';
                    if(index<0 || index>2) {
                        std::cout << "Index of complex number can be 1,2, or 3" << std::endl;
                        return true;
                    }
                    CHECK_MISSING("complex number (z)");
                    config->vars.z[index] = parse_complex(argv[++i]);
                    if(std::isnan(config->vars.z[index].real()) || std::isnan(config->vars.z[index].imag())) {
                        std::cerr << "Malformed complex number format of 'z" << index + 1 << '\'' <<std::endl;
                        return true;
                    }
                    break;
                }
                case '\0':
                default:
                    INVALID_OPTION(argv[i]+1);
            }
        }
    }
    return false;
}