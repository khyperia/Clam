#include "helper.h"
#include <stdexcept>
#include <sstream>
#include <cmath>

void HandleErrImpl(unsigned int errcode, const char* message, const char* filename, int line)
{
    if (errcode != 0)
    {
        std::ostringstream output;
        output << filename;
        output << "(";
        output << line;
        output << "): errcode ";
        output << errcode;
                output << " -- ";
                output << message;
        throw std::runtime_error(output.str());
    }
}

void hsvToRgb(float h, float s, float v, float& r, float& g, float& b)
{
    int i = static_cast<int>(floorf(h * 6));
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch(i % 6)
    {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
}
