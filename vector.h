#pragma once

#ifdef __device__
#define vec_func __device__
#else
#define vec_func
#endif

template<typename T>
vec_func T clamp(T value, T minimum, T maximum)
{
    if (value < minimum)
    {
        return minimum;
    }
    if (value > maximum)
    {
        return maximum;
    }
    return value;
}

template<typename T>
struct Vector2
{
    T x;
    T y;
    vec_func Vector2(T _x, T _y) : x(_x), y(_y)
    {
    }

    vec_func Vector2() : x(0), y(0)
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
struct Vector3
{
    T x;
    T y;
    T z;
    vec_func
    Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z)
    {
    }

    vec_func
    Vector3() : x(0), y(0), z(0)
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
        return sqrtf(length2());
    }

    vec_func
    Vector3<T> normalized() const
    {
        return *this * rsqrtf(length2());
    }

    vec_func
    Vector3<T> clamp(T min, T max) const
    {
        return Vector3<T>(::clamp(x, min, max), ::clamp(y, min, max), ::clamp(z, min, max));
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
    return Vector3<T>(l.y * r.z - l.z * r.y, l.z * r.x - l.x * r.z, l.x * r.y - l.y * r.x);
}

template<typename T>
vec_func Vector3<T> rotate(const Vector3<T> &v, const Vector3<T> &axis, T amount)
{
    return cross(axis, v) * amount + v;
}

template<typename T>
struct Vector4
{
    T x;
    T y;
    T z;
    T w;
    vec_func
    Vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w)
    {
    }

    vec_func
    Vector4() : x(0), y(0), z(0), w(0)
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
        return Vector4<T>(
            ::clamp(x, min, max), ::clamp(y, min, max), ::clamp(z, min, max), ::clamp(w, min, max));
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
