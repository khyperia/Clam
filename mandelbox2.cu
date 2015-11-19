#include <../samples/common/inc/helper_math.h>
#include <float.h>
#include "kernelStructs.h"
#include "mandelbox.h"

struct Mandelbox2State
{
    float4 color;
    float3 position;
    float3 direction;
    float3 normal;
    float3 lightColor;
    int state;
    int traceIter;
    float totalDistance;
};

void Test()
{
    static_assert(sizeof(Mandelbox2State) != Mandelbox2StateSize, "Bad static assert");
}

#define PREVIEW_STATE -1
#define INIT_STATE 0
#define TRACE_STATE 1
#define NORMAL_STATE 2
#define FINISH_STATE 3

static const float NORMAL_DELTA = FLT_EPSILON * 8;

__constant__ MandelboxCfg MandelboxCfgArr[1];
#define cfg (MandelboxCfgArr[0])

__constant__ GpuCameraSettings CameraArr[1];
#define Camera (CameraArr[0])

__constant__ Mandelbox2State* BufferScratchArr[1];
#define BufferScratch (BufferScratchArr[0])

__constant__ uint2* BufferRandArr[1];
#define BufferRand (BufferRandArr[0])

static __device__ float Gauss(float a, float c, float w, float x)
{
    return a * expf(-((x - c) * (x - c)) / (float)(2 * w * w));
}

static __device__ float3 xyz(float4 val)
{
    return make_float3(val.x, val.y, val.z);
}

static __device__ float length2(float2 v)
{
    return dot(v, v);
}

static __device__ float length2(float3 v)
{
    return dot(v, v);
}

static __device__ uint MWC64X(ulong *state)
{
    uint c = (*state) >> 32, x = (*state) & 0xFFFFFFFF;
    *state = x * ((ulong)4294883355U) + c;
    return x ^ c;
}

static __device__ float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / 4294967296.0f;
}

static __device__ float2 RandCircle(ulong* rand)
{
    float2 polar = make_float2(Rand(rand) * 6.28318531f, sqrt(Rand(rand)));
    return make_float2(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

// Box-Muller transform
// returns two normally-distributed independent variables
static __device__ float2 RandNormal(ulong* rand)
{
    float mul = sqrt(-2 * log2(Rand(rand)));
    float angle = 6.28318530718f * Rand(rand);
    return mul * make_float2(cos(angle), sin(angle));
}

static __device__ float3 RandSphere(ulong* rand)
{
    float2 normal;
    float rest;
    do
    {
        normal = RandNormal(rand);
        rest = RandNormal(rand).x;
    } while (normal.x == 0 && normal.y == 0 && rest == 0);
    return normalize(make_float3(normal.x, normal.y, rest));
}

static __device__ float3 RandHemisphere(ulong *rand, float3 normal)
{
    float3 result = RandSphere(rand);
    if (dot(result, normal) < 0)
        result = -result;
    return result;
}

static __device__ ulong GetRand(int x, int y, int width, int frame)
{
    ulong rand;
    uint2 randBuffer = BufferRand[y * width + x];
    rand = (ulong)randBuffer.x << 32 | (ulong)randBuffer.y;
    rand += y * width + x;
    if (frame < 1)
    {
        for (int i = 0; (float)i / cfg.RandSeedInitSteps - 1 < Rand(&rand) * cfg.RandSeedInitSteps;
                i++)
        {
        }
    }
    return rand;
}

static __device__ void SetRand(int x, int y, int width, ulong rand)
{
    BufferRand[y * width + x] = make_uint2((uint)(rand >> 32), (uint)rand);
}

// http://en.wikipedia.org/wiki/Stereographic_projection
static __device__ float3 RayDir(float3 forward, float3 up, float2 screenCoords, float fov)
{
    screenCoords *= -fov;
    float len2 = length2(screenCoords);
    float3 look = make_float3(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);

    float3 right = cross(forward, up);

    return look.x * right + look.y * up + look.z * forward;
}

static __device__ void ApplyDof(float3* position, float3* lookat, float focalPlane, ulong* rand)
{
    float3 focalPosition = *position + *lookat * focalPlane;
    float3 xShift = cross(make_float3(0, 0, 1), *lookat);
    float3 yShift = cross(*lookat, xShift);
    float2 offset = RandCircle(rand);
    float dofPickup = cfg.DofAmount;
    *lookat = normalize(*lookat + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    *position = focalPosition - *lookat * focalPlane;
}

static __device__ void GetCamera(float3 *origin, float3 *direction, int x, int y, int screenX,
        int screenY, int width, int height, ulong *rand)
{
    *origin = make_float3(Camera.posX, Camera.posY, Camera.posZ);
    float3 look = make_float3(Camera.lookX, Camera.lookY, Camera.lookZ);
    float3 up = make_float3(Camera.upX, Camera.upY, Camera.upZ);
    float2 screenCoords = make_float2((float)(x + screenX), (float)(y + screenY));
    screenCoords += make_float2(Rand(rand) - 0.5f, Rand(rand) - 0.5f);
    float fov = Camera.fov * 2 / (width + height);
    //fov *= exp((*hue - 0.5f) * cfg.FovAbberation);
    *direction = RayDir(look, up, screenCoords, fov);
    ApplyDof(origin, direction, Camera.focalDistance, rand);
}

static __device__ float3 Rotate(float3 v, float3 axis, float angle)
{
    float sina = sin(angle);
    float cosa = cos(angle);
    return cosa * v + sina * cross(axis, v) + (1 - cosa) * dot(axis, v) * axis;
}

static __device__ float4 Mandelbulb(float4 z, const float Power)
{
    const float r = length(xyz(z));

    // convert to polar coordinates
    float theta = asin(z.z / r);
    float phi = atan2(z.y, z.x);
    float dr = powf(r, Power - 1.0) * Power * z.w + 1.0;

    // scale and rotate the point
    float zr = pow(r, Power);
    theta = theta * Power;
    phi = phi * Power;

    // convert back to cartesian coordinates
    float3 z3 = zr * make_float3(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
    return make_float4(z3, dr);
}

static __device__ float4 Boxfold(float4 z, const float FoldingLimit)
{
    float3 znew = xyz(z);
    znew = clamp(znew, -FoldingLimit, FoldingLimit) * 2.0f - znew;
    return make_float4(znew, z.w);
}

static __device__ float4 Spherefold(float4 z, const float MinRadius2, const float FixedRadius2)
{
    z *= FixedRadius2 / clamp(length2(xyz(z)), MinRadius2, FixedRadius2);
    return z;
}

static __device__ float4 TScale(float4 z, const float Scale)
{
    return z * make_float4(Scale, Scale, Scale, fabs(Scale));
}

static __device__ float4 TOffset(float4 z, float3 offset)
{
    return z + make_float4(offset, 1.0f);
}

static __device__ float4 TRotate(float4 v, float3 axis, float angle)
{
    return make_float4(Rotate(xyz(v), axis, angle), v.w);
}

static __device__ float4 Mandelbox(float4 z, float3 offset, const float FoldingLimit,
        const float MinRadius2, const float FixedRadius2, const float Scale)
{
    z = Boxfold(z, FoldingLimit);
    z = Spherefold(z, MinRadius2, FixedRadius2);
    z = TScale(z, Scale);
    z = TOffset(z, offset);
    return z;
}

static __device__ float4 DeIter(float4 z, int i, float3 offset, const float FoldingLimit,
        const float MinRadius2, const float FixedRadius2, const float Scale, const float Rotation)
{
    z = Mandelbox(z, offset, FoldingLimit, MinRadius2, FixedRadius2, Scale);
    //const float3 rotVec = normalize(make_float3(1, 1, 1));
    //z = TRotate(z, rotVec, Rotation);
    return z;
}

static __device__ float De(float3 offset)
{
    float4 z = make_float4(offset, 1.0f);
    const float FoldingLimit = cfg.FoldingLimit;
    const float MinRadius2 = cfg.MinRadius2;
    const float FixedRadius2 = cfg.FixedRadius2;
    const float Scale = cfg.Scale;
    const float DeRotation = cfg.DeRotation;
    const float Bailout = cfg.Bailout;
    int n = cfg.MaxIters;
    do
    {
        z = DeIter(z, n, offset, FoldingLimit, MinRadius2, FixedRadius2, Scale, DeRotation);
    } while (length2(xyz(z)) < Bailout && --n);
    return length(xyz(z)) / z.w;
}

static __device__ float DeColor(float3 offset, float lightHue)
{
    float4 z = make_float4(offset, 1.0f);
    float hue = 0;
    const float FoldingLimit = cfg.FoldingLimit;
    const float MinRadius2 = cfg.MinRadius2;
    const float FixedRadius2 = cfg.FixedRadius2;
    const float Scale = cfg.Scale;
    const float DeRotation = cfg.DeRotation;
    const int MaxIters = cfg.MaxIters;
    const float Bailout = cfg.Bailout;
    int n = 0;
    do
    {
        float4 oldz = z;
        z = DeIter(z, n, offset, FoldingLimit, MinRadius2, FixedRadius2, Scale, DeRotation);

        hue += log2(length2(xyz(z) - xyz(oldz)));
    } while (++n < MaxIters && length2(xyz(z)) < Bailout);

    float fullValue = lightHue - hue * cfg.HueVariance;
    fullValue *= 3.14159f;
    fullValue = cos(fullValue);
    fullValue *= fullValue;
    fullValue = pow(fullValue, cfg.ColorSharpness);
    fullValue = 1 - (1 - fullValue) * cfg.Saturation;
    return fullValue;
}

static __device__ float BRDF(float3 normal, float3 incoming, float3 outgoing)
{
    return 1;
    //float3 halfV = normalize(incoming + outgoing);
    //float angle = acos(dot(normal, halfV));
    //return 1 + Gauss(cfg.SpecularHighlightAmount, 0, cfg.SpecularHighlightSize, angle);
}

static __device__ float3 LightBrightness()
{
    float x = Gauss(cfg.LightBrightnessAmount, cfg.LightBrightnessCenter,
    cfg.LightBrightnessWidth, 0.25f);
    float y = Gauss(cfg.LightBrightnessAmount, cfg.LightBrightnessCenter,
    cfg.LightBrightnessWidth, 0.5f);
    float z = Gauss(cfg.LightBrightnessAmount, cfg.LightBrightnessCenter,
    cfg.LightBrightnessWidth, 0.75f);
    return make_float3(x, y, z);
}

static __device__ float3 AmbientBrightness()
{
    float x = Gauss(cfg.AmbientBrightnessAmount, cfg.AmbientBrightnessCenter,
    cfg.AmbientBrightnessWidth, 0.25f);
    float y = Gauss(cfg.AmbientBrightnessAmount, cfg.AmbientBrightnessCenter,
    cfg.AmbientBrightnessWidth, 0.5f);
    float z = Gauss(cfg.AmbientBrightnessAmount, cfg.AmbientBrightnessCenter,
    cfg.AmbientBrightnessWidth, 0.75f);
    return make_float3(x, y, z);
}

static __device__ float SimpleTrace(float3 origin, float3 direction, float width, int height)
{
    const float quality = (width + height) / (2 * Camera.fov);
    float distance = 1.0f;
    float totalDistance = 0.0f;
    int i = 0;
    do
    {
        distance = De(origin + direction * totalDistance) * cfg.DeMultiplier;
        totalDistance += distance;
    } while (++i < cfg.MaxRaySteps && totalDistance < cfg.MaxRayDist
            && distance * quality > totalDistance);
    return (float)(i - 1) / cfg.MaxRaySteps;
}

static __device__ uint PackPixel(float4 pixel)
{
    if (cfg.WhiteClamp)
    {
        float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1)
            pixel /= maxVal;
    }
    pixel = clamp(pixel, 0.0f, 1.0f) * 255;
    return (255 << 24) | ((int)pixel.x << 16) | ((int)pixel.y << 8) | ((int)pixel.z);
}

static __device__ void Finish(Mandelbox2State *state, int x, int y, int width)
{
    float3 finalColor = state->lightColor;
    if (isnan(finalColor.x) || isnan(finalColor.x) || isnan(finalColor.x))
    {
        finalColor = make_float3(1.0f, -1.0f, 1.0f) * 1000000000000.0f;
    }
    float4 oldColor = state->color;
    finalColor = (finalColor + xyz(oldColor) * oldColor.w) / (oldColor.w + 1);
    state->color = make_float4(finalColor, oldColor.w + 1);
}

static __device__ int PreviewState(Mandelbox2State *state, uint* __restrict__ screenPixels, int x,
        int y, int screenX, int screenY, int width, int height, ulong *rand)
{
    GetCamera(&state->position, &state->direction, x, y, screenX, screenY, width, height, rand);
    float value = SimpleTrace(state->position, state->direction, width, height);
    screenPixels[y * width + x] = PackPixel(make_float4(value));
    return PREVIEW_STATE;
}

static __device__ int InitState(Mandelbox2State *state, int x, int y, int screenX, int screenY,
        int width, int height, ulong *rand)
{
    GetCamera(&state->position, &state->direction, x, y, screenX, screenY, width, height, rand);
    state->lightColor = make_float3(1);
    state->totalDistance = 0;
    state->traceIter = 0;
    return TRACE_STATE;
}

static __device__ int TraceState(Mandelbox2State *state, float distance, int width, int height)
{
    state->totalDistance += distance;
    float3 oldPos = state->position;
    state->position += state->direction * distance;
    const float quality = state->traceIter == 0 ?
    cfg.QualityFirstRay * ((width + height) / (2 * Camera.fov)) :
                                                  cfg.QualityRestRay;
    if ((oldPos.x == state->position.x && oldPos.y == state->position.y
            && oldPos.z == state->position.z) || state->totalDistance > cfg.MaxRayDist
            || distance * quality < state->totalDistance)
    {
        state->position = state->position + make_float3(-NORMAL_DELTA, NORMAL_DELTA, NORMAL_DELTA);
        state->normal = make_float3(0, 0, 0);
        if (state->totalDistance > cfg.MaxRayDist)
        {
            float3 mul = AmbientBrightness();
            if (state->direction.x > 0 && state->direction.y > 0 && state->direction.z > 0)
            {
                mul += LightBrightness();
            }
            state->lightColor *= mul;
            return FINISH_STATE;
        }
        return NORMAL_STATE;
    }
    return TRACE_STATE;
}

static __device__ int NormalState(Mandelbox2State *state, float distance, ulong *rand)
{
    if (state->normal.x == 0)
    {
        state->normal.x = distance;
        state->position = state->position
                + (make_float3(NORMAL_DELTA, -NORMAL_DELTA, NORMAL_DELTA)
                        - make_float3(-NORMAL_DELTA, NORMAL_DELTA, NORMAL_DELTA));
    }
    else if (state->normal.y == 0)
    {
        state->normal.y = distance;
        state->position = state->position
                + (make_float3(NORMAL_DELTA, NORMAL_DELTA, -NORMAL_DELTA)
                        - make_float3(NORMAL_DELTA, -NORMAL_DELTA, NORMAL_DELTA));
    }
    else if (state->normal.z == 0)
    {
        state->normal.z = distance;
        state->position = state->position
                + (make_float3(-NORMAL_DELTA, -NORMAL_DELTA, -NORMAL_DELTA)
                        - make_float3(NORMAL_DELTA, NORMAL_DELTA, -NORMAL_DELTA));
    }
    else
    {
        float dnpp = state->normal.x;
        float dpnp = state->normal.y;
        float dppn = state->normal.z;
        float dnnn = distance;

        state->normal = normalize(
                make_float3((dppn + dpnp) - (dnpp + dnnn), (dppn + dnpp) - (dpnp + dnnn),
                        (dpnp + dnpp) - (dppn + dnnn)));
        state->position = state->position
                - make_float3(-NORMAL_DELTA, -NORMAL_DELTA, -NORMAL_DELTA);
        float3 oldRayDir = state->direction;
        state->direction = RandHemisphere(rand, state->normal);
        state->lightColor *= BRDF(state->normal, state->direction, -oldRayDir)
                * dot(state->normal, state->direction);
        state->totalDistance = 0;
        if (++state->traceIter > cfg.NumRayBounces)
        {
            state->position = make_float3(cfg.Bailout + 1);
            return FINISH_STATE;
        }
        return TRACE_STATE;
    }
    return NORMAL_STATE;
}

static __device__ int FinishState(Mandelbox2State *state, uint* __restrict__ screenPixels, int x,
        int y, int width)
{
    Finish(state, x, y, width);
    screenPixels[y * width + x] = PackPixel(state->color);
    return INIT_STATE;
}

extern "C" __global__ void kern(uint* __restrict__ screenPixels, int screenX, int screenY,
        int width, int height, int frame)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    ulong rand = GetRand(x, y, width, frame);

    Mandelbox2State state = BufferScratch[y * width + x];
    if (frame == -1)
    {
        state.state = PREVIEW_STATE;
        state.position = make_float3(cfg.Bailout + 1);
        state.color = make_float4(0);
    }
    else if (frame == 0)
    {
        state.state = INIT_STATE;
        state.position = make_float3(cfg.Bailout + 1);
        state.color = make_float4(0);
        screenPixels[y * width + x] = 255u << 24;
    }

    int i = state.state == PREVIEW_STATE ? 1 : 256;
    do
    {
        float de = De(state.position) * cfg.DeMultiplier;
        switch (state.state)
        {
        case PREVIEW_STATE:
            state.state = PreviewState(&state, screenPixels, x, y, screenX, screenY, width, height,
                    &rand);
            break;
        case INIT_STATE:
            state.state = InitState(&state, x, y, screenX, screenY, width, height, &rand);
            break;
        case TRACE_STATE:
            state.state = TraceState(&state, de, width, height);
            break;
        case NORMAL_STATE:
            state.state = NormalState(&state, de, &rand);
            break;
        case FINISH_STATE:
            state.state = FinishState(&state, screenPixels, x, y, width);
            break;
        }
    } while (--i);

    BufferScratch[y * width + x] = state;
    SetRand(x, y, width, rand);
}
