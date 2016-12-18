#pragma once

#include <iostream>
#include <cmath>

template<typename T>
struct Vector2
{
    T x;
    T y;

    Vector2(T _x, T _y)
        : x(_x), y(_y)
    {
    }

    Vector2()
        : x(0), y(0)
    {
    }

    T length2() const
    {
        return x * x + y * y;
    }

    T length() const
    {
        return std::sqrt(length2());
    }

    Vector2<T> normalized() const
    {
        return *this / length();
    }
};

template<typename T>
Vector2<T> operator+(const Vector2<T> &l, const Vector2<T> &r)
{
    return Vector2<T>(l.x + r.x, l.y + r.y);
}

template<typename T>
Vector2<T> operator-(const Vector2<T> &l, const Vector2<T> &r)
{
    return Vector2<T>(l.x - r.x, l.y - r.y);
}

template<typename T>
Vector2<T> operator-(const Vector2<T> &a)
{
    return Vector2<T>(-a.x, -a.y);
}

template<typename T>
Vector2<T> operator*(const Vector2<T> &l, const T &r)
{
    return Vector2<T>(l.x * r, l.y * r);
}

template<typename T>
Vector2<T> operator*(const T &l, const Vector2<T> &r)
{
    return Vector2<T>(l * r.x, l * r.y);
}

template<typename T>
Vector2<T> operator/(const Vector2<T> &l, const T &r)
{
    return Vector2<T>(l.x / r, l.y / r);
}

template<typename T>
bool operator==(const Vector2<T> &l, Vector2<T> &r)
{
    return l.x == r.x && l.y == r.y;
}

template<typename T>
bool operator!=(const Vector2<T> &l, Vector2<T> &r)
{
    return l.x != r.x || l.y != r.y;
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

    Vector3(T _x, T _y, T _z)
        : x(_x), y(_y), z(_z)
    {
    }

    Vector3()
        : x(0), y(0), z(0)
    {
    }

    T length2() const
    {
        return x * x + y * y + z * z;
    }

    T length() const
    {
        return std::sqrt(length2());
    }

    Vector3<T> normalized() const
    {
        return *this / length();
    }
};

template<typename T>
Vector3<T> operator+(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.x + r.x, l.y + r.y, l.z + r.z);
}

template<typename T>
Vector3<T> operator-(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.x - r.x, l.y - r.y, l.z - r.z);
}

template<typename T>
Vector3<T> operator-(const Vector3<T> &a)
{
    return Vector3<T>(-a.x, -a.y, -a.z);
}

template<typename T>
Vector3<T> operator*(const Vector3<T> &l, const T &r)
{
    return Vector3<T>(l.x * r, l.y * r, l.z * r);
}

template<typename T>
Vector3<T> operator*(const T &l, const Vector3<T> &r)
{
    return Vector3<T>(l * r.x, l * r.y, l * r.z);
}

template<typename T>
Vector3<T> operator/(const Vector3<T> &l, const T &r)
{
    return Vector3<T>(l.x / r, l.y / r, l.z / r);
}

template<typename T>
bool operator==(const Vector3<T> &l, Vector3<T> &r)
{
    return l.x == r.x && l.y == r.y && l.z == r.z;
}

template<typename T>
bool operator!=(const Vector3<T> &l, Vector3<T> &r)
{
    return l.x != r.x || l.y != r.y || l.z != r.z;
}

template<typename T>
Vector3<T> cross(const Vector3<T> &l, const Vector3<T> &r)
{
    return Vector3<T>(l.y * r.z - l.z * r.y,
                      l.z * r.x - l.x * r.z,
                      l.x * r.y - l.y * r.x);
}

template<typename T>
Vector3<T> rotate(const Vector3<T> &v, const Vector3<T> &axis, T amount)
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
