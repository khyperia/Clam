#include <cmath>

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

float acospi(float x)
{
    return (float)acos(x) / M_PI;
}

float cospi(float x)
{
    return (float)cos(x * M_PI);
}

float sinpi(float x)
{
    return (float)sin(x * M_PI);
}

float sqrt(float x)
{
    return std::sqrt(x);
}

float clamp(float x, float low, float high)
{
    if (x < low)
    {
        return low;
    }
    if (x > high)
    {
        return high;
    }
    return x;
}

float3 clamp(float3 x, float low, float high)
{
    float3 res;
    res.x = clamp(x.x, low, high);
    res.y = clamp(x.y, low, high);
    res.z = clamp(x.z, low, high);
    return res;
}

float3 cross(float3 a, float3 b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float fabs(float x)
{
    if (x < 0)
    {
        return -x;
    }
    return x;
}

float length(float3 a)
{
    return sqrt(dot(a, a));
}

float3 normalize(float3 a)
{
    return a * (1.0f / length(a));
}

float log(float a)
{
    return std::log(a);
}

float max(float a, float b)
{
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

int max(int a, int b)
{
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

float min(float a, float b)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

float3 max(float3 x, float3 y)
{
    float3 res;
    res.x = max(x.x, y.x);
    res.y = max(x.y, y.y);
    res.z = max(x.z, y.z);
    return res;
}

float native_rsqrt(float a)
{
    return 1 / sqrt(a);
}

float rootn(float x, int y)
{
    return (float)pow(x, 1.0f / y);
}
