#pragma once

#include <iostream>
#include <cmath>

#ifdef __device__
#define vec_func __device__
#else
#define vec_func
#endif

template<typename T>
vec_func T clamp(T value, T minimum, T maximum)
{
    if (value < minimum)
        return minimum;
    if (value > maximum)
        return maximum;
    return value;
}

template<typename T>
struct Dual
{
    T real;
    T dual;

    vec_func
    Dual(T _real, T _dual)
        : real(_real), dual(_dual)
    {
    }
};

namespace std
{
template<typename T>
vec_func Dual<T> sqrt(const Dual<T> &val)
{
    const T result = sqrtf(val.real);
    return Dual<T>(result, 0.5f / result * val.dual);
}
}

template<typename T>
vec_func Dual<T> sqrtf(const Dual<T> &val)
{
    return std::sqrt(val);
}

template<typename T>
vec_func Dual<T> operator+(const Dual<T> &l, const Dual<T> &r)
{
    return Dual<T>(l.real + r.real, l.dual + r.dual);
}

template<typename T>
vec_func Dual<T> operator-(const Dual<T> &l, const Dual<T> &r)
{
    return Dual<T>(l.real - r.real, l.dual - r.dual);
}

template<typename T>
vec_func Dual<T> operator*(const Dual<T> &l, const Dual<T> &r)
{
    return Dual<T>(l.real * r.real, l.real * r.dual + r.real * l.dual);
}

template<typename T>
vec_func Dual<T> operator/(const Dual<T> &l, const Dual<T> &r)
{
    return Dual<T>(l.real / r.real,
                   (r.real * l.dual - l.real * r.dual) / (r.real * r.real));
}

template<typename T>
struct Vector2
{
    T x;
    T y;

    vec_func Vector2(T _x, T _y)
        : x(_x), y(_y)
    {
    }

    vec_func Vector2()
        : x(0), y(0)
    {
    }

    vec_func T length2() const
    {
        return x * x + y * y;
    }

    vec_func T length() const
    {
        return std::sqrt(length2());
    }

    vec_func Vector2<T> normalized() const
    {
        return *this / length();
    }
};

template<typename T>
vec_func Vector2<T> operator+(const Vector2<T> &l, const Vector2<T> &r)
{
    return Vector2<T>(l.x + r.x, l.y + r.y);
}

template<typename T>
vec_func Vector2<T> &operator+=(Vector2<T> &a, const Vector2<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename T>
vec_func Vector2<T> operator-(const Vector2<T> &l, const Vector2<T> &r)
{
    return Vector2<T>(l.x - r.x, l.y - r.y);
}

template<typename T>
vec_func Vector2<T> &operator-=(Vector2<T> &a, const Vector2<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename T>
vec_func Vector2<T> operator-(const Vector2<T> &a)
{
    return Vector2<T>(-a.x, -a.y);
}

template<typename T>
vec_func Vector2<T> operator*(const Vector2<T> &l, const T &r)
{
    return Vector2<T>(l.x * r, l.y * r);
}

template<typename T>
vec_func Vector2<T> operator*(const T &l, const Vector2<T> &r)
{
    return Vector2<T>(l * r.x, l * r.y);
}

template<typename T>
vec_func Vector2<T> &operator*=(Vector2<T> &a, const T &b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

template<typename T>
vec_func Vector2<T> operator/(const Vector2<T> &l, const T &r)
{
    return Vector2<T>(l.x / r, l.y / r);
}

template<typename T>
vec_func bool operator==(const Vector2<T> &l, const Vector2<T> &r)
{
    return l.x == r.x && l.y == r.y;
}

template<typename T>
vec_func bool operator!=(const Vector2<T> &l, const Vector2<T> &r)
{
    return l.x != r.x || l.y != r.y;
}

template<typename T>
vec_func T dot(const Vector2<T> &l, const Vector2<T> &r)
{
    return l.x * r.x + l.y * r.y;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector2<T> &obj)
{
    os << obj.x << "," << obj.y;
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, Vector2<T> &obj)
{
    char comma;
    is >> obj.x >> comma >> obj.y;
    if (comma != ',')
        is.setstate(std::ios::failbit);
    return is;
}

template<typename T>
struct Vector3
{
    T x;
    T y;
    T z;

    vec_func
    Vector3(T _x, T _y, T _z)
        : x(_x), y(_y), z(_z)
    {
    }

    vec_func
    Vector3()
        : x(0), y(0), z(0)
    {
    }

    vec_func
    T length2() const
    {
        return x * x + y * y + z * z;
    }

    vec_func
    T length() const
    {
        return std::sqrt(length2());
    }

    vec_func
    Vector3<T> normalized() const
    {
        return *this / length();
    }

    vec_func
    Vector3<T> clamp(T min, T max) const
    {
        return Vector3<T>(::clamp(x, min, max),
                          ::clamp(y, min, max),
                          ::clamp(z, min, max));
    }
};

template<typename T>
vec_func Vector3<T> operator+(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.x + r.x, l.y + r.y, l.z + r.z);
}

template<typename T>
vec_func Vector3<T> &operator+=(Vector3<T> &a, const Vector3<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<typename T>
vec_func Vector3<T> operator-(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.x - r.x, l.y - r.y, l.z - r.z);
}

template<typename T>
vec_func Vector3<T> &operator-=(Vector3<T> &a, const Vector3<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<typename T>
vec_func Vector3<T> operator-(const Vector3<T> &a)
{
    return Vector3<T>(-a.x, -a.y, -a.z);
}

template<typename T>
vec_func Vector3<T> operator*(const Vector3<T> &l, const T &r)
{
    return Vector3<T>(l.x * r, l.y * r, l.z * r);
}

template<typename T>
vec_func Vector3<T> operator*(const T &l, const Vector3<T> &r)
{
    return Vector3<T>(l * r.x, l * r.y, l * r.z);
}

template<typename T>
vec_func Vector3<T> &operator*=(Vector3<T> &a, const T &b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

template<typename T>
vec_func Vector3<T> comp_mul(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.x * r.x, l.y * r.y, l.z * r.z);
}

template<typename T>
vec_func Vector3<T> operator/(const Vector3<T> &l, const T &r)
{
    return Vector3<T>(l.x / r, l.y / r, l.z / r);
}

template<typename T>
vec_func bool operator==(const Vector3<T> &l, const Vector3<T> &r)
{
    return l.x == r.x && l.y == r.y && l.z == r.z;
}

template<typename T>
vec_func bool operator!=(const Vector3<T> &l, const Vector3<T> &r)
{
    return l.x != r.x || l.y != r.y || l.z != r.z;
}

template<typename T>
vec_func T dot(const Vector3<T> &l, const Vector3<T> &r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

template<typename T>
vec_func Vector3<T> cross(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.y * r.z - l.z * r.y,
                      l.z * r.x - l.x * r.z,
                      l.x * r.y - l.y * r.x);
}

template<typename T>
vec_func Vector3<T>
rotate(const Vector3<T> &v, const Vector3<T> &axis, T amount)
{
    return cross(axis, v) * amount + v;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector3<T> &obj)
{
    os << obj.x << "," << obj.y << "," << obj.z;
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, Vector3<T> &obj)
{
    char comma1, comma2;
    is >> obj.x >> comma1 >> obj.y >> comma2 >> obj.z;
    if (comma1 != ',' || comma2 != ',')
        is.setstate(std::ios::failbit);
    return is;
}

template<typename T>
struct Vector4
{
    T x;
    T y;
    T z;
    T w;

    vec_func
    Vector4(T _x, T _y, T _z, T _w)
        : x(_x), y(_y), z(_z), w(_w)
    {
    }

    vec_func
    Vector4()
        : x(0), y(0), z(0), w(0)
    {
    }

    vec_func
    T length2() const
    {
        return x * x + y * y + z * z + w * w;
    }

    vec_func
    T length() const
    {
        return std::sqrt(length2());
    }

    vec_func
    Vector4<T> normalized() const
    {
        return *this / length();
    }

    vec_func
    Vector4<T> clamp(T min, T max) const
    {
        return Vector4<T>(::clamp(x, min, max),
                          ::clamp(y, min, max),
                          ::clamp(z, min, max),
                          ::clamp(w, min, max));
    }
};

template<typename T>
vec_func Vector4<T> operator+(const Vector4<T> &l, const Vector4<T> &r)
{
    return Vector4<T>(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
}

template<typename T>
vec_func Vector4<T> &operator+=(Vector4<T> &a, const Vector4<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

template<typename T>
vec_func Vector4<T> operator-(const Vector4<T> &l, const Vector4<T> &r)
{
    return Vector4<T>(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
}

template<typename T>
vec_func Vector4<T> operator-(const Vector4<T> &a)
{
    return Vector4<T>(-a.x, -a.y, -a.z, -a.w);
}

template<typename T>
vec_func Vector4<T> &operator-=(Vector4<T> &a, const Vector4<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

template<typename T>
vec_func Vector4<T> operator*(const Vector4<T> &l, const T &r)
{
    return Vector4<T>(l.x * r, l.y * r, l.z * r, l.w * r);
}

template<typename T>
vec_func Vector4<T> operator*(const T &l, const Vector4<T> &r)
{
    return Vector4<T>(l * r.x, l * r.y, l * r.z, l * r.w);
}

template<typename T>
vec_func Vector4<T> &operator*=(Vector4<T> &a, const T &b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

template<typename T>
vec_func Vector4<T> operator/(const Vector4<T> &l, const T &r)
{
    return Vector4<T>(l.x / r, l.y / r, l.z / r, l.w / r);
}

template<typename T>
vec_func bool operator==(const Vector4<T> &l, const Vector4<T> &r)
{
    return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w;
}

template<typename T>
vec_func bool operator!=(const Vector4<T> &l, const Vector4<T> &r)
{
    return l.x != r.x || l.y != r.y || l.z != r.z || l.w != r.w;
}

template<typename T>
vec_func T dot(const Vector4<T> &l, const Vector4<T> &r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}

template<typename T>
vec_func Vector4<T> comp_mul(const Vector4<T> &l, const Vector4<T> &r)
{
    return Vector4<T>(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector4<T> &obj)
{
    os << obj.x << "," << obj.y << "," << obj.z << "," << obj.w;
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, Vector4<T> &obj)
{
    char comma1, comma2, comma3;
    is >> obj.x >> comma1 >> obj.y >> comma2 >> obj.z >> comma3 >> obj.w;
    if (comma1 != ',' || comma2 != ',' || comma3 != ',')
        is.setstate(std::ios::failbit);
    return is;
}
