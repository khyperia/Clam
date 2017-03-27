// for ide
#ifndef __constant__
#define __constant__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif

#include "vector.h"

struct MandelboxState
{
    Vector4<float> color;
};

struct MandelboxCfg
{
    MandelboxState* scratch;
    uint2* randbuf;
    float posX;
    float posY;
    float posZ;
    float lookX;
    float lookY;
    float lookZ;
    float upX;
    float upY;
    float upZ;
    float fov;
    float focalDistance;
    float Scale;
    float FoldingLimit;
    float FixedRadius2;
    float MinRadius2;
    float DofAmount;
    float LightPosX;
    float LightPosY;
    float LightPosZ;
    int WhiteClamp;
    float LightBrightnessHue;
    float LightBrightnessSat;
    float LightBrightnessVal;
    float AmbientBrightnessHue;
    float AmbientBrightnessSat;
    float AmbientBrightnessVal;
    float ReflectBrightnessHue;
    float ReflectBrightnessSat;
    float ReflectBrightnessVal;
    int MaxIters;
    float Bailout;
    float DeMultiplier;
    int RandSeedInitSteps;
    float MaxRayDist;
    int MaxRaySteps;
    int NumRayBounces;
    float QualityFirstRay;
    float QualityRestRay;
};

__constant__ MandelboxCfg CfgArr;
#define cfg (CfgArr)

/*
namespace staticassert
{
    template<bool x>
    struct test;
    template<>
    struct test<true>
    {
    };
    template<int x, int y>
    struct eq
    {
        test<x == y> foo;
    };
};

static __device__ void Test()
{
    staticassert::eq<sizeof(MandelboxState), MandelboxStateSize> bar;
    (void)bar.foo;
}
*/

static __device__ Vector3<float> xyz(Vector4<float> val)
{
    return Vector3<float>(val.x, val.y, val.z);
}

struct Random
{
    unsigned long long seed;
    __device__ Random(uint2 *randBufferArr, int idx, bool init)
    {
        uint2 randBuffer = randBufferArr[idx];
        seed = (unsigned long long)randBuffer.x << 32 | (unsigned long long)randBuffer.y;
        seed += idx;
        if (init)
        {
            for (int i = 0;
                 (float)i / cfg.RandSeedInitSteps - 1 < Next() * cfg.RandSeedInitSteps;
                 i++)
            {
            }
        }
    }

    __device__ void Save(uint2 *randBufferArr, int idx) const
    {
        randBufferArr[idx] = make_uint2((unsigned int)(seed >> 32), (unsigned int)seed);
    }

    __device__ unsigned int MWC64X()
    {
        unsigned int c = seed >> 32, x = seed & 0xFFFFFFFF;
        seed = x * ((unsigned long long)4294883355U) + c;
        return x ^ c;
    }

    __device__ float Next()
    {
        return (float)MWC64X() / 4294967296.0f;
    }

    __device__ Vector2<float> Circle()
    {
        Vector2<float> polar = Vector2<float>(Next() * 6.28318531f, sqrtf(Next()));
        return Vector2<float>(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
    }

    __device__ Vector2<float> Normal()
    {
        // Box-Muller transform
        // returns two normally-distributed independent variables
        float mul = sqrtf(-2 * log2(Next()));
        float angle = 6.28318530718f * Next();
        return mul * Vector2<float>(cos(angle), sin(angle));
    }

    __device__ Vector3<float> Sphere()
    {
        Vector2<float> temp = Normal();
        Vector3<float> result = Vector3<float>(temp.x, temp.y, Normal().x);
        if (result.length2() != 0)
        {
            return result.normalized();
        }
        else
        {
            return Vector3<float>(1, 0, 0);
        }
    }

    // __device__ Vector3<float> Hemisphere(Vector3<float> normal)
    // {
    //     Vector3<float> result = Sphere();
    //     result *= dot(result, normal) > 0 ? 1.0f : -1.0f;
    //     return result;
    // }
};

struct Ray
{
    Vector3<float> pos;
    Vector3<float> dir;
    __device__ Ray(Vector3<float> pos, Vector3<float> dir) : pos(pos), dir(dir)
    {
    }

    // http://en.wikipedia.org/wiki/Stereographic_projection
    static __device__ Vector3<float>
    RayDir(Vector3<float> forward, Vector3<float> up, Vector2<float> screenCoords, float fov)
    {
        screenCoords *= -fov;
        float len2 = screenCoords.length2();
        Vector3<float> look =
            Vector3<float>(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);
        Vector3<float> right = cross(forward, up);
        return look.x * right + look.y * up + look.z * forward;
    }

    __device__ Vector3<float> At(float time)
    {
        return pos + dir * time;
    }

    __device__ void Dof(float focalPlane, Random &rand)
    {
        Vector3<float> focalPosition = At(focalPlane);
        Vector3<float> xShift = cross(Vector3<float>(0, 0, 1), dir);
        Vector3<float> yShift = cross(dir, xShift);
        Vector2<float> offset = rand.Circle();
        float dofPickup = cfg.DofAmount;
        dir = (dir + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift).normalized();
        pos = focalPosition - dir * focalPlane;
    }

    static __device__ Ray
    Camera(int x, int y, int screenX, int screenY, int width, int height, Random &rand)
    {
        Vector3<float> origin = Vector3<float>(cfg.posX, cfg.posY, cfg.posZ);
        Vector3<float> look = Vector3<float>(cfg.lookX, cfg.lookY, cfg.lookZ);
        Vector3<float> up = Vector3<float>(cfg.upX, cfg.upY, cfg.upZ);
        Vector2<float> screenCoords = Vector2<float>((float)(x + screenX), (float)(y + screenY));
        screenCoords += Vector2<float>(rand.Next() - 0.5f, rand.Next() - 0.5f);
        float fov = cfg.fov * 2 / (width + height);
        Vector3<float> direction = RayDir(look, up, screenCoords, fov);
        Ray result(origin, direction);
        result.Dof(cfg.focalDistance, rand);
        return result;
    }
};

static __device__ Vector3<float> HueToRGB(float hue, float saturation, float value)
{
    hue *= 3;
    float frac = fmod(hue, 1.0f);
    Vector3<float> color;
    switch ((int)hue)
    {
        case 0:
            color = Vector3<float>(1 - frac, frac, 0);
            break;
        case 1:
            color = Vector3<float>(0, 1 - frac, frac);
            break;
        case 2:
            color = Vector3<float>(frac, 0, 1 - frac);
            break;
        default:
            color = Vector3<float>(1, 1, 1);
            break;
    }
    saturation = value * (1 - saturation);
    color = color * (value - saturation) + Vector3<float>(saturation, saturation, saturation);
    return color;
}

struct Fractal
{
    static __device__ Vector4<float> Mandelbulb(Vector4<float> z, const float Power)
    {
        const float r = xyz(z).length();

        // convert to polar coordinates
        float theta = asin(z.z / r);
        float phi = atan2(z.y, z.x);
        float dr = powf(r, Power - 1.0) * Power * z.w + 1.0;

        // scale and rotate the point
        float zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;

        // convert back to cartesian coordinates
        Vector3<float> z3 =
            zr * Vector3<float>(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
        return Vector4<float>(z3.x, z3.y, z3.z, dr);
    }

    static __device__ Vector4<float> BoxfoldD(Vector4<float> z)
    {
        Vector3<float> znew = xyz(z);
        znew = znew.clamp(-cfg.FoldingLimit, cfg.FoldingLimit) * 2.0f - znew;
        return Vector4<float>(znew.x, znew.y, znew.z, z.w);
    }

    static __device__ Vector4<float> ContBoxfoldD(Vector4<float> z)
    {
        Vector3<float> znew = xyz(z);
        Vector3<float> zsq = Vector3<float>(znew.x * znew.x, znew.y * znew.y, znew.z * znew.z);
        zsq += Vector3<float>(1, 1, 1);
        Vector3<float> res = Vector3<float>(
            znew.x / sqrtf(zsq.x), znew.y / sqrtf(zsq.y), znew.z / sqrtf(zsq.z));
        res *= sqrtf(8);
        res = znew - res;
        return Vector4<float>(res.x, res.y, res.z, z.w);
    }

    static __device__ Vector4<float> SpherefoldD(Vector4<float> z)
    {
        z *= cfg.FixedRadius2 / clamp(xyz(z).length2(), cfg.MinRadius2, cfg.FixedRadius2);
        return z;
    }

    static __device__ Vector4<float> ContSpherefoldD(Vector4<float> z)
    {
        z *= cfg.MinRadius2 / xyz(z).length2() + cfg.FixedRadius2;
        return z;
    }

    static __device__ Vector4<float> TScaleD(Vector4<float> z)
    {
        const float scale = cfg.Scale;
        const Vector4<float> mul(scale, scale, scale, fabs(scale));
        return comp_mul(z, mul);
    }

    static __device__ Vector4<float> TOffsetD(Vector4<float> z, Vector3<float> offset)
    {
        return z + Vector4<float>(offset.x, offset.y, offset.z, 1.0f);
    }

    static __device__ Vector4<float> MandelboxD(Vector4<float> z, Vector3<float> offset)
    {
        // z = ContBoxfoldD(z);
        // z = ContSpherefoldD(z);
        z = BoxfoldD(z);
        z = SpherefoldD(z);
        z = TScaleD(z);
        z = TOffsetD(z, offset);
        return z;
    }

    static __device__ float De(Vector3<float> offset)
    {
        Vector4<float> z = Vector4<float>(offset.x, offset.y, offset.z, 1.0f);
        int n = cfg.MaxIters;
        if (n < 1)
        {
            n = 1;
        }
        do
        {
            z = MandelboxD(z, offset);
        }
        while (xyz(z).length2() < cfg.Bailout && --n);
        return xyz(z).length() / z.w;
    }

    static __device__ float Cast(
        Ray ray, const float quality, const float maxDist, Vector3<float> *normal
    )
    {
        float distance;
        float totalDistance = 0.0f;
        int i = 0;
        const int maxSteps = cfg.MaxRaySteps;
        const float deMultiplier = cfg.DeMultiplier;
        do
        {
            distance = De(ray.At(totalDistance)) * deMultiplier;
            totalDistance += distance;
            if (++i == maxSteps)
            {
                *normal = Vector3<float>(1, 0, 0);
                return totalDistance;
            }
        }
        while (totalDistance < maxDist && distance * quality > totalDistance);
        Vector3<float> final = ray.At(totalDistance);
        float delta = 1e-6f; // aprox. 8.3x float epsilon
        if (distance * 0.5f > delta)
        {
            delta = distance * 0.5f;
        }
        float dnpp = De(final + Vector3<float>(-delta, delta, delta));
        float dpnp = De(final + Vector3<float>(delta, -delta, delta));
        float dppn = De(final + Vector3<float>(delta, delta, -delta));
        float dnnn = De(final + Vector3<float>(-delta, -delta, -delta));
        *normal = Vector3<float>((dppn + dpnp) - (dnpp + dnnn),
            (dppn + dnpp) - (dpnp + dnnn),
            (dpnp + dnpp) - (dppn + dnnn));
        if (normal->length2() == 0)
        {
            *normal = Vector3<float>(1, 0, 0);
        }
        *normal = normal->normalized();
        return totalDistance;
    }
};

struct Tracer
{
    static __device__ Vector3<float> AmbientBrightness()
    {
        return HueToRGB(cfg.AmbientBrightnessHue, cfg.AmbientBrightnessSat, cfg.AmbientBrightnessVal
        );
    }

    static __device__ Vector3<float> LightBrightness()
    {
        return HueToRGB(cfg.LightBrightnessHue, cfg.LightBrightnessSat, cfg.LightBrightnessVal);
    }

    static __device__ Vector3<float> ReflectBrightness()
    {
        return HueToRGB(cfg.ReflectBrightnessHue, cfg.ReflectBrightnessSat, cfg.ReflectBrightnessVal);
    }

    static __device__ Vector3<float> LightPos()
    {
        return Vector3<float>(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    }

    static __device__ float
    Specular(Vector3<float> incoming, Vector3<float> outgoing, Vector3<float> normal)
    {
        const Vector3<float> half = (incoming + outgoing).normalized();
        const float dot_prod = fabsf(dot(half, normal));
        const float hardness = 128.0f;
        //const float base_reflection = 0.7f; // TODO: Make configurable.
        const float base_reflection = cfg.ReflectBrightnessVal;
        const float spec = powf(sinpif(dot_prod * 0.5f), hardness);
        return spec * (1 - base_reflection) + base_reflection;
    }

    static __device__ Vector3<float> Trace(Ray ray, int width, int height, Random &rand)
    {
        Vector3<float> rayColor(0, 0, 0);
        float reflectionColor = 1.0f;
        bool firstRay = true;
        const float maxDist = cfg.MaxRayDist;
        for (int i = 0; i < cfg.NumRayBounces; i++)
        {
            const float quality = firstRay ? cfg.QualityFirstRay *
                ((width + height) / (2 * cfg.fov)) : cfg.QualityRestRay;
            Vector3<float> normal;
            const float distance = Fractal::Cast(ray, quality, maxDist, &normal);
            if (distance > maxDist)
            {
                Vector3<float> color = firstRay
                    ? Vector3<float>(0.3, 0.3, 0.3)
                    : AmbientBrightness();
                rayColor += color * reflectionColor;
                break;
            }
            // incorporates lambertian lighting
            const Vector3<float> newDir = (rand.Sphere() + normal).normalized();
            const Vector3<float> newPos = ray.At(distance);

            // direct lighting
            // TODO: Soft shadows, shuffle lightPos a bit
            const Vector3<float> toLightVec = LightPos() - newPos;
            const float lightDist = toLightVec.length();
            const Vector3<float> toLight = toLightVec * (1 / lightDist);
            float normalDotProd = dot(normal, toLight);
            if (normalDotProd > 0)
            {
                const float dimmingFactor =
                    (normalDotProd * reflectionColor * Specular(toLight, -ray.dir, normal)) /
                        (lightDist * lightDist);
                const float distance = Fractal::Cast(
                    Ray(newPos, toLight), quality, lightDist, &normal
                );
                if (distance >= lightDist)
                {
                    rayColor += LightBrightness() * dimmingFactor;
                }
            }
            reflectionColor *= Specular(newDir, -ray.dir, normal);
            ray = Ray(newPos, newDir);
            firstRay = false;
        }
        return rayColor;
    }
};

static __device__ unsigned int PackPixel(Vector3<float> pixel)
{
    const float gamma_correct = 1 / 2.2f;
    pixel.x = powf(pixel.x, gamma_correct);
    pixel.y = powf(pixel.y, gamma_correct);
    pixel.z = powf(pixel.z, gamma_correct);
    if (cfg.WhiteClamp)
    {
        float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1)
        {
            pixel *= 1.0f / maxVal;
        }
    }
    else
    {
        pixel = pixel.clamp(0.0f, 1.0f);
    }
    pixel = comp_mul(pixel, Vector3<float>(255, 255, 255));
    return ((unsigned int)255 << 24) | ((unsigned int)(unsigned char)pixel.x << 16) |
        ((unsigned int)(unsigned char)pixel.y << 8) | ((unsigned int)(unsigned char)pixel.z);
}

// type: -1 is preview, 0 is init, 1 is continue
extern "C" __global__ void Main(
    unsigned int *__restrict__ screenPixels,
    int screenX,
    int screenY,
    int width,
    int height,
    int frame
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (y >= height)
    {
        return;
    }
    // flip image - in screen space, 0,0 is top-left, in 3d space, 0,0 is bottom-left
    y = height - (y + 1);
    Random rand(static_cast<uint2 *>(cfg.randbuf), idx, frame <= 0);
    Ray ray = Ray::Camera(x, y, screenX, screenY, width, height, rand);
    Vector3<float> color = Tracer::Trace(ray, width, height, rand);
    MandelboxState *scratch = &cfg.scratch[idx];
    Vector4<float> oldColor = frame > 0 ? scratch->color : Vector4<float>(0, 0, 0, 0);
    float newWeight = oldColor.w + 1;
    Vector3<float> newColor = (color + xyz(oldColor) * oldColor.w) / newWeight;
    scratch->color = Vector4<float>(newColor.x, newColor.y, newColor.z, newWeight);

    int packedColor = PackPixel(newColor);
    screenPixels[idx] = packedColor;
    rand.Save(cfg.randbuf, idx);
}
