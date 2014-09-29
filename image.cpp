#include "image.h"
#include <string>
#include <limits>
#include <stdexcept>
#include <stdlib.h>
extern "C"
{
#include <png.h>
}

inline uint8_t convByte(float value)
{
    const uint8_t maxval = std::numeric_limits<uint8_t>::max();
    return static_cast<uint8_t>(std::max(std::min(value * maxval,
                    static_cast<float>(maxval)), 0.0f));
}

void WriteImage(std::vector<float> const& float4Pixels, unsigned long width)
{
    auto homePtr = getenv("HOME");
    if (!homePtr)
    {
        puts("Cannot save image: $HOME not set");
    }
    std::string imageFilename = homePtr;
    imageFilename += "/fractal.png";

    FILE* fp = fopen(imageFilename.c_str(), "wb");
    if (!fp)
        throw std::runtime_error("Could not open " + imageFilename + " for writing");

    auto png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
        throw std::runtime_error("Could not create png write struct");

    auto info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        throw std::runtime_error("Could not create png info struct");

    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("Error during init_io");

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("Error during writing of header");
    
    unsigned long height = float4Pixels.size() / (4 * width);
    png_set_IHDR(png_ptr, info_ptr,
            static_cast<unsigned int>(width), static_cast<unsigned int>(height),
            8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("Error during writing bytes");

    std::vector<uint8_t> row_ptr(width * 3);
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            row_ptr[col * 3 + 0] = convByte(float4Pixels[(row * width + col) * 4 + 0]);
            row_ptr[col * 3 + 1] = convByte(float4Pixels[(row * width + col) * 4 + 1]);
            row_ptr[col * 3 + 2] = convByte(float4Pixels[(row * width + col) * 4 + 2]);
        }
        png_write_row(png_ptr, row_ptr.data());
    }

    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("Error during end of write");

    png_write_end(png_ptr, nullptr);

    fclose(fp);

    puts("Saved image:");
    puts(imageFilename.c_str());
}
