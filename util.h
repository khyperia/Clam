#pragma once

#include <sstream>
#include <cuda.h>
#include <iostream>
#include <GL/gl.h>

#define HandleCu(x) HandleCuImpl(x, #x, __FILE__, __LINE__)

static inline void HandleCuImpl(CUresult err, const char* expr, const char* file, const int line)
{
    if (err != CUDA_SUCCESS)
    {
        std::ostringstream msg;
        const char* errstr;
        if (cuGetErrorString(err, &errstr) != CUDA_SUCCESS)
        {
            errstr = "Unknown error";
        }
        msg << "CUDA Error (" << err << "): " << errstr << std::endl;
        msg << file << "(" << line << "): " << expr;
        throw std::runtime_error(msg.str());
    }
}

template<typename T>
static std::string tostring(const T &value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template<typename T>
static T fromstring(const std::string &value)
{
    std::istringstream stream(value);
    T result;
    stream >> result;
    return result;
}

static inline void HandleGl()
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        const GLubyte *str = glGetString(err);
        if (str == NULL)
        {
            throw std::runtime_error("OpenGL error no string: " + tostring(err));
        }
        throw std::runtime_error(std::string("OpenGL error: ") + (const char *)str);
    }
}
