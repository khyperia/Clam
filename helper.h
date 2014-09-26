#pragma once

#include <cmath>

template<typename T>
struct is_pointer { static const bool value = false; };

template<typename T>
struct is_pointer<T*> { static const bool value = true; };

void HandleErrImpl(int errcode, const char* message, const char* filename, int line);

#define HandleErr(errcode) HandleErrImpl((errcode), (#errcode), __FILE__, __LINE__)

void hsvToRgb(float h, float s, float v, float& r, float& g, float& b);
