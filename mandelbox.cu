#include "vector.h"
#include <float.h>

typedef void *CUdeviceptr;

#include "kernelStructs.h"

struct MandelboxState
{
    Vector4<float> color;
    Vector4<float> lightColor;
    Vector3<float> position;
    Vector3<float> direction;
    Vector3<float> normal;
    int state;
    int traceIter;
    float totalDistance;
};
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

#define PREVIEW_STATE -1
#define INIT_STATE 0
#define TRACE_STATE 1
#define NORMAL_STATE 2
#define SUBSURFACE_STATE 3
#define FINISH_STATE 4

#define NORMAL_DELTA (FLT_EPSILON * 8)

__constant__ MandelboxCfg CfgArr[1];

#define cfg (CfgArr[0])
#define BufferScratch ((MandelboxState*)cfg.scratch)
#define BufferRand ((uint2 *)cfg.randbuf)

// static __device__ float Gauss(float a, float c, float w, float x)
// {
//     return a * expf(-((x - c) * (x - c)) / (float)(2 * w * w));
// }

static __device__ Vector3<float> xyz(Vector4<float> val)
{
    return Vector3<float>(val.x, val.y, val.z);
}

static __device__ uint MWC64X(unsigned long long *state)
{
    uint c = (*state) >> 32, x = (*state) & 0xFFFFFFFF;
    *state = x * ((unsigned long long)4294883355U) + c;
    return x ^ c;
}

static __device__ float Rand(unsigned long long *seed)
{
    return (float)MWC64X(seed) / 4294967296.0f;
}

static __device__ Vector2<float> RandCircle(unsigned long long *rand)
{
    Vector2<float>
        polar = Vector2<float>(Rand(rand) * 6.28318531f, sqrt(Rand(rand)));
    return Vector2<float>(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

// Box-Muller transform
// returns two normally-distributed independent variables
static __device__ Vector2<float> RandNormal(unsigned long long *rand)
{
    float mul = sqrt(-2 * log2(Rand(rand)));
    float angle = 6.28318530718f * Rand(rand);
    return mul * Vector2<float>(cos(angle), sin(angle));
}

static __device__ Vector3<float> RandSphere(unsigned long long *rand)
{
    Vector2<float> normal;
    float rest;
    do
    {
        normal = RandNormal(rand);
        rest = RandNormal(rand).x;
    }
    while (normal.x == 0 && normal.y == 0 && rest == 0);
    return Vector3<float>(normal.x, normal.y, rest).normalized();
}

static __device__ Vector3<float>
RandHemisphere(unsigned long long *rand, Vector3<float> normal)
{
    Vector3<float> result = RandSphere(rand);
    if (dot(result, normal) < 0)
        result = -result;
    return result;
}

static __device__ unsigned long long GetRand(int x, int y, int width, bool init)
{
    unsigned long long rand;
    uint2 randBuffer = BufferRand[y * width + x];
    rand = (unsigned long long)randBuffer.x << 32
        | (unsigned long long)randBuffer.y;
    rand += y * width + x;
    if (init)
    {
        for (int i = 0; (float)i / cfg.RandSeedInitSteps - 1
            < Rand(&rand) * cfg.RandSeedInitSteps; i++)
        {
        }
    }
    return rand;
}

static __device__ void SetRand(int x, int y, int width, unsigned long long rand)
{
    BufferRand[y * width + x] = make_uint2((uint)(rand >> 32), (uint)rand);
}

// http://en.wikipedia.org/wiki/Stereographic_projection
static __device__ Vector3<float> RayDir(Vector3<float> forward,
                                        Vector3<float> up,
                                        Vector2<float> screenCoords,
                                        float fov)
{
    screenCoords *= -fov;
    float len2 = screenCoords.length2();
    Vector3<float> look =
        Vector3<float>(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1)
            / -(len2 + 1);

    Vector3<float> right = cross(forward, up);

    return look.x * right + look.y * up + look.z * forward;
}

static __device__ void ApplyDof(Vector3<float> *position,
                                Vector3<float> *lookat,
                                float focalPlane,
                                unsigned long long *rand)
{
    Vector3<float> focalPosition = *position + *lookat * focalPlane;
    Vector3<float> xShift = cross(Vector3<float>(0, 0, 1), *lookat);
    Vector3<float> yShift = cross(*lookat, xShift);
    Vector2<float> offset = RandCircle(rand);
    float dofPickup = cfg.DofAmount;
    *lookat = (*lookat + offset.x * dofPickup * xShift
        + offset.y * dofPickup * yShift).normalized();
    *position = focalPosition - *lookat * focalPlane;
}

static __device__ void GetCamera(Vector3<float> *origin,
                                 Vector3<float> *direction,
                                 int x,
                                 int y,
                                 int screenX,
                                 int screenY,
                                 int width,
                                 int height,
                                 unsigned long long *rand)
{
    *origin = Vector3<float>(cfg.posX, cfg.posY, cfg.posZ);
    Vector3<float> look = Vector3<float>(cfg.lookX, cfg.lookY, cfg.lookZ);
    Vector3<float> up = Vector3<float>(cfg.upX, cfg.upY, cfg.upZ);
    Vector2<float> screenCoords =
        Vector2<float>((float)(x + screenX), (float)(y + screenY));
    screenCoords += Vector2<float>(Rand(rand) - 0.5f, Rand(rand) - 0.5f);
    float fov = cfg.fov * 2 / (width + height);
    //fov *= exp((*hue - 0.5f) * cfg.FovAbberation);
    *direction = RayDir(look, up, screenCoords, fov);
    ApplyDof(origin, direction, cfg.focalDistance, rand);
}

static __device__ Vector3<float>
HueToRGB(float hue, float saturation, float value)
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
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    saturation = value * (1 - saturation);
    color = color * (value - saturation)
        + Vector3<float>(saturation, saturation, saturation);
    return color;
}

static __device__ Vector3<float> LightBrightness()
{
    return HueToRGB(cfg.LightBrightnessHue,
                    cfg.LightBrightnessSat,
                    cfg.LightBrightnessVal);
}

static __device__ Vector3<float> AmbientBrightness()
{
    return HueToRGB(cfg.AmbientBrightnessHue,
                    cfg.AmbientBrightnessSat,
                    cfg.AmbientBrightnessVal);
}

template<typename T>
struct NumTraits;

template<>
struct NumTraits<float>
{
    static __device__ float Const(const float value)
    {
        return value;
    }
};

template<>
struct NumTraits<Dual<float> >
{
    static __device__ Dual<float> Const(const float value)
    {
        return Dual<float>(value, 0);
    }
};

template<typename T>
struct Fractal
{
    static __device__ Vector3<T> Rotate(Vector3<T> v, Vector3<T> axis, T angle)
    {
        T sina = sin(angle);
        T cosa = cos(angle);
        return cosa * v + sina * cross(axis, v)
            + (1 - cosa) * dot(axis, v) * axis;
    }

    static __device__ Vector4<T> Mandelbulb(Vector4<T> z, const T Power)
    {
        const T r = xyz(z).length();

        // convert to polar coordinates
        T theta = asin(z.z / r);
        T phi = atan2(z.y, z.x);
        T dr = powf(r, Power - 1.0) * Power * z.w + 1.0;

        // scale and rotate the point
        T zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;

        // convert back to cartesian coordinates
        Vector3<T> z3 = zr * Vector3<T>(cos(theta) * cos(phi),
                                        cos(theta) * sin(phi),
                                        sin(theta));
        return Vector4<T>(z3.x, z3.y, z3.z, dr);
    }

    static __device__ Vector4<T> BoxfoldD(Vector4<T> z)
    {
        Vector3<T> znew = xyz(z);
        znew = znew.clamp(-cfg.FoldingLimit, cfg.FoldingLimit) * 2.0f - znew;
        return Vector4<T>(znew.x, znew.y, znew.z, z.w);
    }

    static __device__ Vector4<T> ContBoxfoldD(Vector4<T> z)
    {
        Vector3<T> znew = xyz(z);
        Vector3<T>
            zsq = Vector3<T>(znew.x * znew.x, znew.y * znew.y, znew.z * znew.z);
        zsq += Vector3<T>(NumTraits<T>::Const(1),
                          NumTraits<T>::Const(1),
                          NumTraits<T>::Const(1));
        Vector3<T> res = Vector3<T>(znew.x / sqrtf(zsq.x),
                                    znew.y / sqrtf(zsq.y),
                                    znew.z / sqrtf(zsq.z));
        res *= NumTraits<T>::Const(sqrtf(8));
        res = znew - res;
        return Vector4<T>(res.x, res.y, res.z, z.w);
    }

    static __device__ Vector3<T> ContBoxfold(Vector3<T> z)
    {
        Vector3<T> zsq = Vector3<T>(z.x * z.x + NumTraits<T>::Const(1),
                                    z.y * z.y + NumTraits<T>::Const(1),
                                    z.z * z.z + NumTraits<T>::Const(1));
        Vector3<T>
            res(z.x / sqrtf(zsq.x), z.y / sqrtf(zsq.y), z.z / sqrtf(zsq.z));
        return z - res * NumTraits<T>::Const(sqrtf(8));
    }

    static __device__ Vector4<T> SpherefoldD(Vector4<T> z)
    {
        z *= cfg.FixedRadius2
            / clamp(xyz(z).length2(), cfg.MinRadius2, cfg.FixedRadius2);
        return z;
    }

    static __device__ Vector4<T> ContSpherefoldD(Vector4<T> z)
    {
        z *= NumTraits<T>::Const(cfg.MinRadius2) / xyz(z).length2()
            + NumTraits<T>::Const(cfg.FixedRadius2);
        return z;
    }

    static __device__ Vector3<T> ContSpherefold(Vector3<T> z)
    {
        const T fixedRadius2 = NumTraits<T>::Const(cfg.FixedRadius2);
        const Vector3<T>
            fixedRadius2_vec(fixedRadius2, fixedRadius2, fixedRadius2);
        return z * NumTraits<T>::Const(cfg.MinRadius2) / z.length2()
            + fixedRadius2_vec;
    }

    static __device__ Vector4<T> TScaleD(Vector4<T> z)
    {
        const T scale = NumTraits<T>::Const(cfg.Scale);
        const Vector4<T> mul(scale, scale, scale, fabs(scale));
        return comp_mul(z, mul);
    }

    static __device__ Vector3<T> TScale(Vector3<T> z)
    {
        const T scale = NumTraits<T>::Const(cfg.Scale);
        const Vector3<T> mul(scale, scale, scale);
        return comp_mul(z, mul);
    }

    static __device__ Vector4<T> TOffsetD(Vector4<T> z, Vector3<T> offset)
    {
        return z + Vector4<T>(offset.x, offset.y, offset.z, 1.0f);
    }

    static __device__ Vector3<T> TOffset(Vector3<T> z, Vector3<T> offset)
    {
        return z + Vector3<T>(offset.x, offset.y, offset.z);
    }

    static __device__ Vector4<T>
    TRotateD(Vector4<T> v, Vector3<T> axis, T angle)
    {
        const Vector3<T> rot = Rotate(xyz(v), axis, angle);
        return Vector4<T>(rot.x, rot.y, rot.z, v.w);
    }

    static __device__ Vector4<T> MandelboxD(Vector4<T> z, Vector3<T> offset)
    {
        // z = ContBoxfoldD(z);
        // z = ContSpherefoldD(z);
        z = BoxfoldD(z);
        z = SpherefoldD(z);
        z = TScaleD(z);
        z = TOffsetD(z, offset);
        return z;
    }

    static __device__ Vector3<T> Mandelbox(Vector3<T> z, Vector3<T> offset)
    {
        z = ContBoxfold(z);
        z = ContSpherefold(z);
        z = TScale(z);
        z = TOffset(z, offset);
        return z;
    }

    static __device__ Vector4<T> DeIterD(Vector4<T> z, int i, Vector3<T> offset)
    {
        z = MandelboxD(z, offset);
        // Rotation adds a *significant* amount of instructions to the DE, and slows down execution considerably.
        //const Vector3<T> rotVec = normalize(Vector3<T>(cfg.DeRotationAxisX, cfg.DeRotationAxisY, cfg.DeRotationAxisZ));
        //z = TRotate(z, rotVec, cfg.DeRotationAmount);
        return z;
    }
};

static __device__ float
DeDir(Vector3<float> offset, Vector3<float> direction, Vector3<float> *lighting)
{
    Vector3<Dual<float> > dual(Dual<float>(offset.x, direction.x),
                               Dual<float>(offset.y, direction.y),
                               Dual<float>(offset.z, direction.z));
    Vector3<Dual<float> > dual_offset = dual;
    int n = cfg.MaxIters;
    do
    {
        dual = Fractal<Dual<float> >::Mandelbox(dual, dual_offset);
    }
    while (dual.length2().real < cfg.Bailout && --n);

    float length_real = dual.x.real * dual.x.real + dual.y.real * dual.y.real
        + dual.z.real * dual.z.real;
    float length_dual = dual.x.dual * dual.x.dual + dual.y.dual * dual.y.dual
        + dual.z.dual * dual.z.dual;

    float distance = sqrtf(length_real) / sqrtf(length_dual) * 0.25f;
    const Vector3<float>
        lightPos = Vector3<float>(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float lightDistance = (lightPos - offset).length() - cfg.LightSize;
    if (lightDistance < distance)
    {
        *lighting = LightBrightness();
        return lightDistance;
    }
    else
    {
        *lighting = Vector3<float>(0, 0, 0);
        return distance;
    }
}

static __device__ float De(Vector3<float> offset, Vector3<float> *lighting)
{
    Vector4<float> z = Vector4<float>(offset.x, offset.y, offset.z, 1.0f);
    int n = cfg.MaxIters;
    do
    {
        z = Fractal<float>::DeIterD(z, n, offset);
    }
    while (xyz(z).length2() < cfg.Bailout && --n);
    float distance = xyz(z).length() / z.w;
    const Vector3<float>
        lightPos = Vector3<float>(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float lightDistance = (lightPos - offset).length() - cfg.LightSize;
    if (lightDistance < distance)
    {
        *lighting = LightBrightness();
        return lightDistance;
    }
    else
    {
        *lighting = Vector3<float>(0, 0, 0);
        return distance;
    }
}

static __device__ Vector3<float>
RayToLight(Vector3<float> position, float *probability, unsigned long long *rand)
{
    Vector3<float>
        lightPos = Vector3<float>(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    // hyp = d
    // opposite = r
    // theta = asin(r / d)
    // area of spherical cap = 2 * pi * r^2 * (1 - cos(theta))
    // area = 2 * pi * (1 - cos(asin(r / d)))
    // probability = 2 * (1 - cos(asin(r / d))) / 4
    // simplified = (1 - sqrt(1 - r^2/d^2)) / 2
    float dist = (lightPos - position).length();
    dist = fmax(dist, cfg.LightSize);
    float costheta = sqrtf(1 - (cfg.LightSize * cfg.LightSize) / (dist * dist));
    *probability = (1 - costheta) / 2;

    Vector3<float> lightPosR = lightPos + RandSphere(rand) * cfg.LightSize;
    Vector3<float> lightDirR = (lightPosR - position).normalized();

    return lightDirR;
}

static __device__ Vector3<float> NewRayDir(Vector3<float> position,
                                           Vector3<float> normal,
                                           Vector3<float> oldRayDir,
                                           float *probability,
                                           unsigned long long *rand)
{
    float prob;
    Vector3<float> lightDirR = RayToLight(position, &prob, rand);
    if (dot(lightDirR, normal) > 0)
    {
        if (Rand(rand) > 0.5f)
        {
            *probability *= prob;
            return lightDirR;
        }
        *probability *= 1 - prob;
    }
    return RandHemisphere(rand, normal);
}

static __device__ float
BRDF(Vector3<float> normal, Vector3<float> incoming, Vector3<float> outgoing)
{
    return 1;
    //Vector3<float> halfV = normalize(incoming + outgoing);
    //float angle = acos(dot(normal, halfV));
    //return Gauss(1, 0, cfg.SpecularHighlightSize, angle);
}

static __device__ float SimpleTrace(Vector3<float> origin,
                                    Vector3<float> direction,
                                    float width,
                                    int height)
{
    const float quality = (width + height) / (2 * cfg.fov);
    float distance = 1.0f;
    float totalDistance = 0.0f;
    int i = 0;
    do
    {
        Vector3<float> lighting;
        distance = De(origin + direction * totalDistance, &lighting)
            * cfg.DeMultiplier;
        totalDistance += distance;
    }
    while (++i < cfg.MaxRaySteps && totalDistance < cfg.MaxRayDist
        && distance * quality > totalDistance);
    return (float)(i - 1) / cfg.MaxRaySteps;
}

static __device__ int Bounce(MandelboxState *state, unsigned long long *rand)
{
    Vector3<float> oldRayDir = state->direction;
    state->direction = NewRayDir(state->position,
                                 state->normal,
                                 oldRayDir,
                                 &state->lightColor.w,
                                 rand);
    Vector3<float> newLightColor = xyz(state->lightColor);
    Vector3<float> multiply_factor =
        BRDF(state->normal, state->direction, -oldRayDir)
            * dot(state->normal, state->direction)
            * HueToRGB(cfg.ReflectHue, cfg.ReflectSat, cfg.ReflectVal);
    newLightColor = comp_mul(newLightColor, multiply_factor);
    state->lightColor = Vector4<float>(newLightColor.x,
                                       newLightColor.y,
                                       newLightColor.z,
                                       state->lightColor.w);
    state->totalDistance = 0;

    if (++state->traceIter > cfg.NumRayBounces)
    {
        const float outside = cfg.Bailout + 1;
        state->position = Vector3<float>(outside, outside, outside);
        return FINISH_STATE;
    }
    return TRACE_STATE;
}

static __device__ uint PackPixel(Vector3<float> pixel)
{
    if (cfg.WhiteClamp)
    {
        float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1)
            pixel *= 1.0f / maxVal;
    }
    pixel = comp_mul(pixel.clamp(0.0f, 1.0f), Vector3<float>(255, 255, 255));
    return (255 << 24) | ((int)pixel.x << 16) | ((int)pixel.y << 8)
        | ((int)pixel.z);
}

static __device__ void Finish(MandelboxState *state, int x, int y, int width)
{
    Vector4<float> finalColor = state->lightColor;
    if (isnan(finalColor.x) || isnan(finalColor.x) || isnan(finalColor.x))
    {
        //finalColor = Vector4<float>(1.0f, -1.0f, 1.0f, 1.0f) * 1000.0f;
        finalColor = Vector4<float>(0.0f, 0.0f, 0.0f, 1.0f) * 0.001f;
    }
    Vector4<float> oldColor = state->color;
    Vector3<float> result =
        (xyz(finalColor) * finalColor.w + xyz(oldColor) * oldColor.w)
            / (oldColor.w + finalColor.w);
    state->color =
        Vector4<float>(result.x, result.y, result.z, oldColor.w + finalColor.w);
}

static __device__ int PreviewState(MandelboxState *state,
                                   uint *__restrict__ screenPixels,
                                   int x,
                                   int y,
                                   int screenX,
                                   int screenY,
                                   int width,
                                   int height,
                                   unsigned long long *rand)
{
    GetCamera(&state->position,
              &state->direction,
              x,
              y,
              screenX,
              screenY,
              width,
              height,
              rand);
    float value = SimpleTrace(state->position, state->direction, width, height);
    value = sqrtf(value);
    // TODO: for some reason the image is flipped vertically (SDL?)
    int index = (height - y - 1) * width + x;
    screenPixels[index] = PackPixel(Vector3<float>(value, value, value));
    return PREVIEW_STATE;
}

static __device__ int InitState(MandelboxState *state,
                                int x,
                                int y,
                                int screenX,
                                int screenY,
                                int width,
                                int height,
                                unsigned long long *rand)
{
    GetCamera(&state->position,
              &state->direction,
              x,
              y,
              screenX,
              screenY,
              width,
              height,
              rand);
    state->lightColor = Vector4<float>(1, 1, 1, 1);
    state->totalDistance = 0;
    state->traceIter = 0;
    return TRACE_STATE;
}

static __device__ int TraceState(MandelboxState *state,
                                 float distance,
                                 Vector3<float> lighting,
                                 int width,
                                 int height)
{
    state->totalDistance += distance;
    Vector3<float> oldPos = state->position;
    state->position += state->direction * distance;
    const float quality = state->traceIter == 0 ? cfg.QualityFirstRay
        * ((width + height) / (2 * cfg.fov)) : cfg.QualityRestRay;
    if ((oldPos.x == state->position.x && oldPos.y == state->position.y
        && oldPos.z == state->position.z)
        || state->totalDistance > cfg.MaxRayDist
        || distance * quality < state->totalDistance)
    {
        state->position = state->position
            + Vector3<float>(-NORMAL_DELTA, NORMAL_DELTA, NORMAL_DELTA);
        state->normal = Vector3<float>(0, 0, 0);
        if (state->totalDistance > cfg.MaxRayDist)
        {
            if (state->traceIter == 0)
            {
                lighting = Vector3<float>(0.3, 0.3, 0.3);
            }
            else
            {
                lighting = AmbientBrightness();
            }
        }
        if (lighting.length2() > 0)
        {
            Vector3<float>
                newColor = comp_mul(xyz(state->lightColor), lighting);
            state->lightColor = Vector4<float>(newColor.x,
                                               newColor.y,
                                               newColor.z,
                                               state->lightColor.w);
            return FINISH_STATE;
        }
        return NORMAL_STATE;
    }
    return TRACE_STATE;
}

static __device__ int NormalState(MandelboxState *state,
                                  float distance,
                                  Vector3<float> lighting,
                                  unsigned long long *rand)
{
    if (lighting.length2() > 0)
    {
        Vector3<float> newColor = comp_mul(xyz(state->lightColor), lighting);
        state->lightColor = Vector4<float>(newColor.x,
                                           newColor.y,
                                           newColor.z,
                                           state->lightColor.w);
        return FINISH_STATE;
    }
    if (state->normal.x == 0)
    {
        state->normal.x = distance;
        state->position = state->position
            + (Vector3<float>(NORMAL_DELTA, -NORMAL_DELTA, NORMAL_DELTA)
                - Vector3<float>(-NORMAL_DELTA, NORMAL_DELTA, NORMAL_DELTA));
    }
    else if (state->normal.y == 0)
    {
        state->normal.y = distance;
        state->position = state->position
            + (Vector3<float>(NORMAL_DELTA, NORMAL_DELTA, -NORMAL_DELTA)
                - Vector3<float>(NORMAL_DELTA, -NORMAL_DELTA, NORMAL_DELTA));
    }
    else if (state->normal.z == 0)
    {
        state->normal.z = distance;
        state->position = state->position
            + (Vector3<float>(-NORMAL_DELTA, -NORMAL_DELTA, -NORMAL_DELTA)
                - Vector3<float>(NORMAL_DELTA, NORMAL_DELTA, -NORMAL_DELTA));
    }
    else
    {
        float dnpp = state->normal.x;
        float dpnp = state->normal.y;
        float dppn = state->normal.z;
        float dnnn = distance;

        state->normal = Vector3<float>((dppn + dpnp) - (dnpp + dnnn),
                                       (dppn + dnpp) - (dpnp + dnnn),
                                       (dpnp + dnpp) - (dppn + dnnn))
            .normalized();
        state->position = state->position
            - Vector3<float>(-NORMAL_DELTA, -NORMAL_DELTA, -NORMAL_DELTA);

        if (Rand(rand) < 0.125f)
        {
            float subsurface_jump_amount =
                (dnpp + dpnp + dppn + dnnn) * (0.25f * 10);
            // technically incorrect, since jumpback is anti-normal, but whatever
            Vector3<float> subsurface_jump = RandHemisphere(rand, -state->normal) * subsurface_jump_amount;
            //Vector3<float> subsurface_jump = state->normal * -subsurface_jump_amount;
            state->position = state->position + subsurface_jump;
            // use as threshhold
            state->totalDistance = subsurface_jump_amount;
            return SUBSURFACE_STATE;
        }

        return Bounce(state, rand);
    }
    return NORMAL_STATE;
}

static __device__ int SubsurfaceState(MandelboxState *state,
                                      float distance,
                                      Vector3<float> lighting,
                                      unsigned long long *rand)
{
    float subsurface_jump_amount = state->totalDistance;
    // bad subsurface test
    if (distance > subsurface_jump_amount
        || distance < subsurface_jump_amount * 0.25f)
    {
        Vector3<float>
            subsurface_jump = state->normal * -subsurface_jump_amount;
        state->position = state->position - subsurface_jump;
        return Bounce(state, rand);
    }

    float subsurface_distance = 1 - distance / subsurface_jump_amount;
    float prob;
    state->direction = RayToLight(state->position, &prob, rand);
    Vector3<float> newLightColor = xyz(state->lightColor);
    Vector3<float> multiply_factor = HueToRGB(0.2f, 0.2f, subsurface_distance);
    newLightColor = comp_mul(newLightColor, multiply_factor);
    state->lightColor = Vector4<float>(newLightColor.x,
                                       newLightColor.y,
                                       newLightColor.z,
                                       state->lightColor.w);

    state->totalDistance = 0;
    ++state->traceIter;
    return TRACE_STATE;
}

static __device__ int FinishState(MandelboxState *state,
                                  uint *__restrict__ screenPixels,
                                  int x,
                                  int y,
                                  int width,
                                  int height)
{
    Finish(state, x, y, width);
    // TODO: for some reason the image is flipped vertically (SDL?)
    int index = (height - y - 1) * width + x;
    screenPixels[index] = PackPixel(xyz(state->color));
    return INIT_STATE;
}

// type: -1 is preview, 0 is init, 1 is continue
extern "C" __global__ void kern(uint *__restrict__ screenPixels,
                                int screenX,
                                int screenY,
                                int width,
                                int height,
                                int type)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    unsigned long long rand = GetRand(x, y, width, type <= 0);

    MandelboxState state = BufferScratch[y * width + x];
    if (type == -1)
    {
        state.state = PREVIEW_STATE;
        const float outside = cfg.Bailout + 1;
        state.position = Vector3<float>(outside, outside, outside);
        state.color = Vector4<float>(0, 0, 0, 0);
    }
    else if (type == 0)
    {
        state.state = INIT_STATE;
        const float outside = cfg.Bailout + 1;
        state.position = Vector3<float>(outside, outside, outside);
        state.color = Vector4<float>(0, 0, 0, 0);
        // TODO: for some reason the image is flipped vertically (SDL?)
        int index = (height - y - 1) * width + x;
        screenPixels[index] = 255u << 24;
    }

    int i = state.state == PREVIEW_STATE ? 1 : max(1 << cfg.ItersPerKernel, 1);
    do
    {
        Vector3<float> lighting;
        float de = De(state.position, &lighting) * cfg.DeMultiplier;
        switch (state.state)
        {
            case PREVIEW_STATE:
                state.state = PreviewState(&state,
                                           screenPixels,
                                           x,
                                           y,
                                           screenX,
                                           screenY,
                                           width,
                                           height,
                                           &rand);
                break;
            case INIT_STATE:
                state.state = InitState(&state,
                                        x,
                                        y,
                                        screenX,
                                        screenY,
                                        width,
                                        height,
                                        &rand);
                break;
            case TRACE_STATE:
                state.state = TraceState(&state, de, lighting, width, height);
                break;
            case NORMAL_STATE:
                state.state = NormalState(&state, de, lighting, &rand);
                break;
            case SUBSURFACE_STATE:
                state.state = SubsurfaceState(&state, de, lighting, &rand);
                break;
            case FINISH_STATE:
                state.state =
                    FinishState(&state, screenPixels, x, y, width, height);
                break;
        }
    }
    while (--i);

    BufferScratch[y * width + x] = state;
    SetRand(x, y, width, rand);
}
