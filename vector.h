#pragma once

#include <cmath>

template<typename T>
struct Vector4
{
    T x;
    T y;
    T z;
    T w;

    Vector4(T _x, T _y, T _z, T _w)
            : x(_x), y(_y), z(_z), w(_w)
    {
    }

    Vector4()
            : x(0), y(0), z(0), w(0)
    {
    }

    T length2()
    {
        return x * x + y * y + z * z + w * w;
    }

    T length()
    {
        return std::sqrt(length2());
    }

    Vector4<T> normalized()
    {
        return *this / length();
    }
};

template<typename T>
Vector4<T> operator+(const Vector4<T> &l, const Vector4<T> &r)
{
    return Vector4<T>(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
}

template<typename T>
Vector4<T> operator-(const Vector4<T> &l, const Vector4<T> &r)
{
    return Vector4<T>(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
}

template<typename T>
Vector4<T> operator-(const Vector4<T> &a)
{
    return Vector4<T>(-a.x, -a.y, -a.z, -a.w);
}

template<typename T>
Vector4<T> operator*(const Vector4<T> &l, const T &r)
{
    return Vector4<T>(l.x * r, l.y * r, l.z * r, l.w * r);
}

template<typename T>
Vector4<T> operator*(const T &l, const Vector4<T> &r)
{
    return Vector4<T>(l * r.x, l * r.y, l * r.z, l * r.w);
}

template<typename T>
Vector4<T> operator/(const Vector4<T> &l, const T &r)
{
    return Vector4<T>(l.x / r, l.y / r, l.z / r, l.w / r);
}

template<typename T>
bool operator==(const Vector4<T> &l, Vector4<T> &r)
{
    return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w;
}

template<typename T>
bool operator!=(const Vector4<T> &l, Vector4<T> &r)
{
    return l.x != r.x || l.y != r.y || l.z != r.z || l.w != r.w;
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

    T length2()
    {
        return x * x + y * y + z * z;
    }

    T length()
    {
        return std::sqrt(length2());
    }

    Vector3<T> normalized()
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
    return Vector3<T>(
            l.y * r.z - l.z * r.y,
            l.z * r.x - l.x * r.z,
            l.x * r.y - l.y * r.x
    );
}

template<typename T>
Vector3<T> rotate(const Vector3<T> &v, const Vector3<T> &axis, T amount)
{
    return cross(axis, v) * amount + v;
}

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

    T length2()
    {
        return x * x + y * y;
    }

    T length()
    {
        return std::sqrt(length2());
    }

    Vector2<T> normalized()
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
