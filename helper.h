#pragma once

#include <memory>

template<typename T>
struct is_pointer { static const bool value = false; };

template<typename T>
struct is_pointer<T*> { static const bool value = true; };

void HandleErrImpl(int errcode, const char* message, const char* filename, int line);

#define HandleErr(errcode) HandleErrImpl((errcode), (#errcode), __FILE__, __LINE__)

void hsvToRgb(float h, float s, float v, float& r, float& g, float& b);

    template<typename T, typename Func, typename... Args>
std::shared_ptr<T> make_custom_shared(Func const destructor, Args... args)
{
    return std::shared_ptr<T>(new T(args...), [destructor](T* toDelete)
        {
            destructor(*toDelete);
            delete toDelete;
        });
}
