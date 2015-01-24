// Note: This is a module using builtin functions of the master program. Edit at your own risk.

#include "../socketHelper.h"
#include "../luaHelper.h"
#include "../master.h"
#include <lauxlib.h>
#include <png.h>

// This section is a nightmare of libpng.

static inline unsigned char convByte(float value)
{
    value = value * UCHAR_MAX;
    if (value < 0)
        value = 0;
    if (value > UCHAR_MAX)
        value = UCHAR_MAX;
    return (unsigned char)value;
}

int savePng(float* data, long width, long height)
{
    const char* home = getenv("HOME");
    if (!home)
    {
        puts("Cannot save image: $HOME not set");
        return -1;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/clam2_buffer.png", home);

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
        fclose(fp);
        return -1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        puts("Could not create png info struct");
        fclose(fp);
        return -1;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during init_io");
        fclose(fp);
        return -1;
    }

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error during writing of header");
        fclose(fp);
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
        fclose(fp);
        return -1;
    }

    unsigned char* row_ptr = malloc_s((size_t)width * 3);
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
        fclose(fp);
        return -1;
    }

    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);

    fclose(fp);

    printf("Saved image %s\n", filename);
    return 0;
}

float* loadPng(const char* filename, long* width, long* height)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        perror(filename);
        return NULL;
    }

    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        printf("%s was not a PNG file\n", filename);
        fclose(fp);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        puts("png_create_read_struct returned null");
        return NULL;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        puts("png_create_info_struct returned null.");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        fclose(fp);
        return NULL;
    }

    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info)
    {
        puts("png_create_info_struct returned 0.");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        fclose(fp);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        puts("Error from libpng.");
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_uint_32 temp_width, temp_height;
    png_get_IHDR(png_ptr, info_ptr, &temp_width, &temp_height, &bit_depth, &color_type,
            NULL, NULL, NULL);

    if (width)
    {*width = temp_width;}
    if (height)
    {*height = temp_height;}

    png_read_update_info(png_ptr, info_ptr);

    png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);

    png_bytepp imageData = malloc_s((size_t)*height * sizeof(png_bytep));
    for (int y = 0; y < *height; y++)
        imageData[y] = malloc_s((size_t)rowbytes * sizeof(png_byte));

    png_read_image(png_ptr, imageData);

    float* ret = malloc_s((size_t)*width * (size_t)*height * 4 * 4);

    for (int y = 0; y < *height; y++)
    {
        for (int x = 0; x < *width; x++)
        {
            ret[(y * *width + x) * 4 + 0] = imageData[y][x * 3 + 0] / 255.0f;
            ret[(y * *width + x) * 4 + 1] = imageData[y][x * 3 + 1] / 255.0f;
            ret[(y * *width + x) * 4 + 2] = imageData[y][x * 3 + 2] / 255.0f;
            ret[(y * *width + x) * 4 + 3] = 0.0f;
        }
    }

    for (int y = 0; y < *height; y++)
        free(imageData[y]);
    free(imageData);

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    fclose(fp);
    return ret;
}

// Lua section

// TODO: Find a better way to get buffer size
static long dlbuffer_callback_width;

int dlbuffer_callback(float* data, long bufferSize)
{
    long width = dlbuffer_callback_width;
    if (PrintErr(savePng(data, width, bufferSize / (width * 4 * (long)sizeof(float)))))
        puts("Ignoring PNG error.");
    return 0;
}

// Downloads a buffer to a png on disk
int run_dlbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageDlBuffer));
    long callback = (long)&dlbuffer_callback;
    LuaPrintErr(send_all(sockets, &callback, sizeof(callback)));
    int bufferId = (int)luaL_checkinteger(state, 1);
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(int)));
    dlbuffer_callback_width = luaL_checkinteger(state, 2);
    return 0;
}

// Uploads a png to a buffer
int run_uplbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageUplBuffer));
    int bufferId = (int)luaL_checkinteger(state, 1);
    const char* filename = luaL_checkstring(state, 2);
    long width = 0, height = 0;
    float* buffer = loadPng(filename, &width, &height);
    if (!buffer)
    {
        LuaPrintErr("loadPng error" && 1);
    }
    long size = width * height * 4 * 4;
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(bufferId)));
    LuaPrintErr(send_all(sockets, &size, sizeof(size)));
    LuaPrintErr(send_all(sockets, buffer, (size_t)size));
    lua_pushnumber(state, width);
    lua_pushnumber(state, height);
    return 2;
}

int luaopen_imagebuffer(lua_State* state)
{
    lua_register(state, "dlbuffer", run_dlbuffer);
    lua_register(state, "uplbuffer", run_uplbuffer);
    return 0;
}
