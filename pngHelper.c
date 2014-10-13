#include <stdlib.h>
#include <stdio.h>
#include <png.h>

inline unsigned char convByte(float value)
{
    value = value * UCHAR_MAX;
    if (value < 0)
        value = 0;
    if (value > UCHAR_MAX)
        value = UCHAR_MAX;
    return (unsigned char)value;
}

int savePng(int uid1, int uid2, float* data, long width, long height)
{
    const char* home = getenv("HOME");
    if (!home)
    {
        puts("Cannot save image: $HOME not set");
        return -1;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/fractal_id%d_sock%d.png", home, uid1, uid2);

    FILE* fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Could not open %s for writing\n", filename);
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        puts("Could not create png write struct");
        return -1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        puts("Could not create png info struct");
        return -1;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during init_io");
        return -1;
    }

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during writing of header");
        return -1;
    }

    png_set_IHDR(png_ptr, info_ptr,
            (png_uint_32)width, (png_uint_32)height,
            8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during writing bytes");
        return -1;
    }

    unsigned char* row_ptr = malloc((size_t)width * 3);
    if (!row_ptr)
    {
        puts("malloc() failed");
        exit(-1);
    }
    for (long row = 0; row < height; row++)
    {
        for (long col = 0; col < width; col++)
        {
            row_ptr[col * 3 + 0] = convByte(data[(row * width + col) * 4 + 0]);
            row_ptr[col * 3 + 1] = convByte(data[(row * width + col) * 4 + 1]);
            row_ptr[col * 3 + 2] = convByte(data[(row * width + col) * 4 + 2]);
        }
        png_write_row(png_ptr, row_ptr);
    }

    free(row_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during end of write");
        return -1;
    }

    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);

    fclose(fp);

    printf("Saved image %s\n", filename);
    return 0;
}
