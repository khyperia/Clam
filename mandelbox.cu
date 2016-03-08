#include <../samples/common/inc/helper_math.h>
#include <float.h>
#include "kernelStructs.h"
#include "mandelbox.h"

struct MandelboxState
{
    float4 color;
    float4 lightColor;
    float3 position;
    float3 direction;
    float3 normal;
    int state;
    int traceIter;
    float totalDistance;
};

void Test()
{
//    static_assert(sizeof(MandelboxState) != MandelboxStateSize, "Bad static assert");
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

__constant__ MandelboxState* BufferScratchArr[1];
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

static __device__ float3 HueToRGB(float hue, float saturation, float value)
{
    hue *= 3;
    float frac = fmod(hue, 1.0f);
    float3 color;
    switch ((int)hue)
    {
    case 0:
        color = make_float3(1 - frac, frac, 0);
        break;
    case 1:
        color = make_float3(0, 1 - frac, frac);
        break;
    case 2:
        color = make_float3(frac, 0, 1 - frac);
        break;
    default:
        color = make_float3(1);
        break;
    }
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    saturation = value * (1 - saturation);
    color = color * (value - saturation) + saturation;
    return color;
}

static __device__ float3 LightBrightness()
{
    return HueToRGB(
        cfg.LightBrightnessHue,
        cfg.LightBrightnessSat,
        cfg.LightBrightnessVal
        );
}

static __device__ float3 AmbientBrightness()
{
    return HueToRGB(
        cfg.AmbientBrightnessHue,
        cfg.AmbientBrightnessSat,
        cfg.AmbientBrightnessVal
        );
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

static __device__ float4 Boxfold(float4 z)
{
    float3 znew = xyz(z);
    znew = clamp(znew, -cfg.FoldingLimit, cfg.FoldingLimit) * 2.0f - znew;
    return make_float4(znew, z.w);
}

static __device__ float4 Spherefold(float4 z)
{
    z *= cfg.FixedRadius2 / clamp(length2(xyz(z)), cfg.MinRadius2, cfg.FixedRadius2);
    return z;
}

static __device__ float4 TScale(float4 z)
{
    return z * make_float4(cfg.Scale, cfg.Scale, cfg.Scale, fabs(cfg.Scale));
}

static __device__ float4 TOffset(float4 z, float3 offset)
{
    return z + make_float4(offset, 1.0f);
}

static __device__ float4 TRotate(float4 v, float3 axis, float angle)
{
    return make_float4(Rotate(xyz(v), axis, angle), v.w);
}

static __device__ float4 Mandelbox(float4 z, float3 offset)
{
    z = Boxfold(z);
    z = Spherefold(z);
    z = TScale(z);
    z = TOffset(z, offset);
    return z;
}

static __device__ float4 DeIter(float4 z, int i, float3 offset)
{
    z = Mandelbox(z, offset);
    const float3 rotVec = normalize(make_float3(cfg.DeRotationAxisX, cfg.DeRotationAxisY, cfg.DeRotationAxisZ));
    z = TRotate(z, rotVec, cfg.DeRotationAmount);
    return z;
}

static __device__ float De(float3 offset, float3 *lighting)
{
    float4 z = make_float4(offset, 1.0f);
    int n = cfg.MaxIters;
    do
    {
        z = DeIter(z, n, offset);
    } while (length2(xyz(z)) < cfg.Bailout && --n);
    float distance = length(xyz(z)) / z.w;
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float lightDistance = length(lightPos - offset) - cfg.LightSize;
    if (lightDistance < distance)
    {
        *lighting = LightBrightness();
        return lightDistance;
    }
    else
    {
        *lighting = make_float3(0);
        return distance;
    }
}

static __device__ float3 NewRayDir(float3 position, float3 normal, float3 oldRayDir,
        float *probability, ulong *rand)
{
    float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
     /*
        hyp = d
        opposite = r
        theta = asin(r / d)
        area of spherical cap = 2 * pi * r^2 * (1 - cos(theta))
        area = 2 * pi * (1 - cos(asin(r / d)))
        probability = 2 * (1 - cos(asin(r / d))) / 4
        simplified = (1 - sqrt(1 - r^2/d^2)) / 2
    */
    float dist = length(lightPos - position);
    dist = fmax(dist, cfg.LightSize);
    float costheta = sqrtf(1 - (cfg.LightSize * cfg.LightSize) / (dist * dist));
    float prob = (1 - costheta) / 2;

    float3 lightPosR = lightPos + RandSphere(rand) * cfg.LightSize;
    float3 lightDirR = normalize(lightPosR - position);
    float3 lightDir = normalize(lightPos - position);
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

static __device__ float BRDF(float3 normal, float3 incoming, float3 outgoing)
{
    return 1;
    //float3 halfV = normalize(incoming + outgoing);
    //float angle = acos(dot(normal, halfV));
    //return Gauss(1, 0, cfg.SpecularHighlightSize, angle);
}

static __device__ float SimpleTrace(float3 origin, float3 direction, float width, int height)
{
    const float quality = (width + height) / (2 * Camera.fov);
    float distance = 1.0f;
    float totalDistance = 0.0f;
    int i = 0;
    do
    {
        float3 lighting;
        distance = De(origin + direction * totalDistance, &lighting) * cfg.DeMultiplier;
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

static __device__ void Finish(MandelboxState *state, int x, int y, int width)
{
    float4 finalColor = state->lightColor;
    if (isnan(finalColor.x) || isnan(finalColor.x) || isnan(finalColor.x))
    {
        finalColor = make_float4(1.0f, -1.0f, 1.0f, 1.0f) * 1000.0f;
    }
    float4 oldColor = state->color;
    float3 result = (xyz(finalColor) * finalColor.w + xyz(oldColor) * oldColor.w)
            / (oldColor.w + finalColor.w);
    state->color = make_float4(result, oldColor.w + finalColor.w);
}

static __device__ int PreviewState(MandelboxState *state, uint* __restrict__ screenPixels, int x,
        int y, int screenX, int screenY, int width, int height, ulong *rand)
{
    GetCamera(&state->position, &state->direction, x, y, screenX, screenY, width, height, rand);
    float value = SimpleTrace(state->position, state->direction, width, height);
    value = sqrtf(value);
    screenPixels[y * width + x] = PackPixel(make_float4(value));
    return PREVIEW_STATE;
}

static __device__ int InitState(MandelboxState *state, int x, int y, int screenX, int screenY,
        int width, int height, ulong *rand)
{
    GetCamera(&state->position, &state->direction, x, y, screenX, screenY, width, height, rand);
    state->lightColor = make_float4(1);
    state->totalDistance = 0;
    state->traceIter = 0;
    return TRACE_STATE;
}

static __device__ int TraceState(MandelboxState *state, float distance, float3 lighting, int width,
        int height)
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
            lighting = AmbientBrightness();
        }
        if (length2(lighting) > 0)
        {
            float3 newColor = xyz(state->lightColor) * lighting;
            state->lightColor = make_float4(newColor, state->lightColor.w);
            return FINISH_STATE;
        }
        return NORMAL_STATE;
    }
    return TRACE_STATE;
}

static __device__ int NormalState(MandelboxState *state, float distance, float3 lighting, ulong *rand)
{
    if (length2(lighting) > 0)
    {
        float3 newColor = xyz(state->lightColor) * lighting;
        state->lightColor = make_float4(newColor, state->lightColor.w);
        return FINISH_STATE;
    }
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
        state->direction = NewRayDir(state->position, state->normal, oldRayDir,
                &state->lightColor.w, rand);
        float3 newLightColor = xyz(state->lightColor);
        newLightColor *= BRDF(state->normal, state->direction, -oldRayDir)
                * dot(state->normal, state->direction);
        state->lightColor = make_float4(newLightColor, state->lightColor.w);
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

static __device__ int FinishState(MandelboxState *state, uint* __restrict__ screenPixels, int x,
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

    MandelboxState state = BufferScratch[y * width + x];
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

    int i = state.state == PREVIEW_STATE ? 1 : max(1 << cfg.ItersPerKernel, 1);
    do
    {
        float3 lighting;
        float de = De(state.position, &lighting) * cfg.DeMultiplier;
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
            state.state = TraceState(&state, de, lighting, width, height);
            break;
        case NORMAL_STATE:
            state.state = NormalState(&state, de, lighting, &rand);
            break;
        case FINISH_STATE:
            state.state = FinishState(&state, screenPixels, x, y, width);
            break;
        }
    } while (--i);

    BufferScratch[y * width + x] = state;
    SetRand(x, y, width, rand);
}
