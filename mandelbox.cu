#include <../samples/common/inc/helper_math.h>
#include <float.h>
#include "kernelStructs.h"
#include "mandelbox.h"

__constant__ MandelboxCfg MandelboxCfgArr[1];
#define cfg (MandelboxCfgArr[0])

__constant__ GpuCameraSettings CameraArr[1];
#define Camera (CameraArr[0])

__constant__ float4* BufferScratchArr[1];
#define BufferScratch (BufferScratchArr[0])

__constant__ uint2* BufferRandArr[1];
#define BufferRand (BufferRandArr[0])

__device__ static float Gauss(float a, float c, float w, float x)
{
	return a * expf(-((x - c) * (x - c)) / (float)(2 * w * w));
}

__device__ static float3 xyz(float4 val)
{
    return make_float3(val.x, val.y, val.z);
}

__device__ static float length2(float2 v)
{
    return dot(v, v);
}

__device__ static float length2(float3 v)
{
    return dot(v, v);
}

// http://en.wikipedia.org/wiki/Stereographic_projection
__device__ static float3 RayDir(float3 forward, float3 up, float2 screenCoords, float fov) {
    screenCoords *= -fov;
    float len2 = length2(screenCoords);
    float3 look = make_float3(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1)
            / -(len2 + 1);

    float3 right = cross(forward, up);

    return look.x * right + look.y * up + look.z * forward;
}

__device__ static float3 Rotate(float3 v, float3 axis, float angle)
{
    float sina = sin(angle);
    float cosa = cos(angle);
    return cosa * v + sina * cross(axis, v) + (1 - cosa) * dot(axis, v) * axis;
}

__device__ static float4 Mandelbulb(float4 z, const float Power)
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

__device__ static float4 Boxfold(float4 z, const float FoldingLimit)
{
    float3 znew = xyz(z);
    znew = clamp(znew, -FoldingLimit, FoldingLimit) * 2.0f - znew;
    return make_float4(znew.x, znew.y, znew.z, z.w);
}

__device__ static float4 Spherefold(float4 z, const float MinRadius2, const float FixedRadius2)
{
    z *= FixedRadius2 / clamp(length2(xyz(z)), MinRadius2, FixedRadius2);
    return z;
}

__device__ static float4 TScale(float4 z, const float Scale)
{
    return z * make_float4(Scale, Scale, Scale, fabs(Scale));
}

__device__ static float4 TOffset(float4 z, float3 offset)
{
    return z + make_float4(offset.x, offset.y, offset.z, 1.0f);
}

__device__ static float4 TRotate(float4 v, float3 axis, float angle)
{
    return make_float4(Rotate(xyz(v), axis, angle), v.w);
}

__device__ static float4 Mandelbox(float4 z, float3 offset,
        const float FoldingLimit,
        const float MinRadius2, const float FixedRadius2,
        const float Scale)
{
    z = Boxfold(z, FoldingLimit);
    z = Spherefold(z, MinRadius2, FixedRadius2);
    z = TScale(z, Scale);
    z = TOffset(z, offset);
    return z;
}

__device__ static float4 DeIter(float4 z, int i,
        float3 offset,
        const float FoldingLimit,
        const float MinRadius2, const float FixedRadius2,
        const float Scale,
        const float Rotation)
{
    z = Mandelbox(z, offset, FoldingLimit, MinRadius2, FixedRadius2, Scale);
    const float3 rotVec = normalize(make_float3(1, 1, 1));
    z = TRotate(z, rotVec, Rotation);
    return z;
}

__device__ static float De(float3 offset) {
    float4 z = make_float4(offset, 1.0f);
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
        z = DeIter(z, n, offset, FoldingLimit, MinRadius2, FixedRadius2, Scale, DeRotation);
    } while (length2(xyz(z)) < Bailout && ++n < MaxIters);
    return length(xyz(z)) / z.w;
}

__device__ static float DeColor(float3 offset, float lightHue) {
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

__device__ static uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

__device__ static float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / 4294967296.0f;
}

__device__ static float2 RandCircle(ulong* rand)
{
    float2 polar = make_float2(Rand(rand) * 6.28318531f, sqrt(Rand(rand)));
    return make_float2(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

// Box-Muller transform
// returns two normally-distributed independent variables
__device__ static float2 RandNormal(ulong* rand)
{
    float mul = sqrt(-2 * log2(Rand(rand)));
    float angle = 6.28318530718f * Rand(rand);
    return mul * make_float2(cos(angle), sin(angle));
}

__device__ static float3 RandSphere(ulong* rand)
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

__device__ static float3 RandHemisphere(ulong* rand, float3 normal)
{
    float3 result = RandSphere(rand);
    if (dot(result, normal) < 0)
        result = -result;
    return result;
}

__device__ static void ApplyDof(float3* position, float3* lookat, float focalPlane, float hue, ulong* rand)
{
    float3 focalPosition = *position + *lookat * focalPlane;
    float3 xShift = cross(make_float3(0, 0, 1), *lookat);
    float3 yShift = cross(*lookat, xShift);
    float2 offset = RandCircle(rand);
    float dofPickup = cfg.DofAmount;
    *lookat = normalize(*lookat + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    *position = focalPosition - *lookat * focalPlane;
}

__device__ static float3 Normal(float3 pos) {
    const float delta = FLT_EPSILON * 2;
    float dppn = De(pos + make_float3(delta, delta, -delta));
    float dpnp = De(pos + make_float3(delta, -delta, delta));
    float dnpp = De(pos + make_float3(-delta, delta, delta));
    float dnnn = De(pos + make_float3(-delta, -delta, -delta));

    return normalize(make_float3(
                (dppn + dpnp) - (dnpp + dnnn),
                (dppn + dnpp) - (dpnp + dnnn),
                (dpnp + dnpp) - (dppn + dnnn)
                ));
}

__device__ static float RaySphereIntersection(float3 rayOrigin, float3 rayDir,
    float3 sphereCenter, float sphereSize,
    bool canBePast)
{
    float3 omC = rayOrigin - sphereCenter;
    float lDotOmC = dot(rayDir, omC);
    float underSqrt = lDotOmC * lDotOmC - dot(omC, omC) + sphereSize * sphereSize;
    if (underSqrt < 0)
        return FLT_MAX;
    float theSqrt = sqrt(underSqrt);
    float dist = -lDotOmC - theSqrt;
    if (dist > 0)
        return dist;
    dist = -lDotOmC + theSqrt;
    if (canBePast && dist > 0)
        return dist;
    return FLT_MAX;
}

__device__ static float Trace(float3 origin, float3 direction, float quality, float hue, ulong* rand,
        int* isFog, int* hitLightsource)
{
    float distance = 1.0f;
    float totalDistance = De(origin) * cfg.DeMultiplier * Rand(rand);
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float sphereDist = RaySphereIntersection(origin, direction, lightPos, cfg.LightSize, false);
    float fogDist = -log2(Rand(rand)) / (float)(cfg.FogDensity /* * hue */);
    float maxRayDist = min(min((float)cfg.MaxRayDist, fogDist), sphereDist);
    int i = 0;
    do
    {
        distance = De(origin + direction * totalDistance) * cfg.DeMultiplier;
        totalDistance += distance;
    } while (++i < cfg.MaxRaySteps && totalDistance < maxRayDist &&
             distance * quality > totalDistance);
    if (totalDistance > sphereDist)
        *hitLightsource = 1;
    else
        *hitLightsource = 0;
    if (totalDistance > fogDist)
    {
        *isFog = 1;
        totalDistance = fogDist;
    }
    else if (totalDistance > cfg.MaxRayDist)
        *isFog = 1;
    else
        *isFog = 0;
    return totalDistance;
}

__device__ static float SimpleTrace(float3 origin, float3 direction, float quality)
{
    float distance = 1.0f;
    float totalDistance = 0.0f;
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float sphereDist = RaySphereIntersection(origin, direction, lightPos, cfg.LightSize, false);
    float maxRayDist = min((float)cfg.MaxRayDist, sphereDist);
    int i = 0;
    do
    {
        distance = De(origin + direction * totalDistance) * cfg.DeMultiplier;
        totalDistance += distance;
    } while (++i < cfg.MaxRaySteps && totalDistance < maxRayDist && distance * quality > totalDistance);
    return (float)(i - 1) / cfg.MaxRaySteps;
}

__device__ static bool Reaches(float3 initial, float3 final)
{
    float3 direction = final - initial;
    float lenDir = length(direction);
    direction /= lenDir;
    float totalDistance = 0;
    float distance = FLT_MAX;
    float threshHold = fabs(De(final)) * (cfg.DeMultiplier * 0.5f);
    int i = 0;
    do
    {
        distance = De(initial + direction * totalDistance) * cfg.DeMultiplier;
        if (i == 0 && fabs(distance * 0.5f) < threshHold)
            threshHold = fabs(distance * 0.5f);
        totalDistance += distance;
        if (totalDistance > lenDir)
            return true;
    } while (++i < cfg.MaxRaySteps && totalDistance < cfg.MaxRayDist && distance > threshHold);
    return false;
}

__device__ static float LightBrightness(float hue)
{
    return Gauss(cfg.LightBrightnessAmount, cfg.LightBrightnessCenter, cfg.LightBrightnessWidth, hue);
}

__device__ static float AmbientBrightness(float hue)
{
    return Gauss(cfg.AmbientBrightnessAmount, cfg.AmbientBrightnessCenter, cfg.AmbientBrightnessWidth, hue);
}

__device__ static float DirectLighting(float3 rayPos, float hue, ulong* rand, float3* lightDir)
{
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float3 lightToRay = normalize(rayPos - lightPos);
    float3 movedLightPos = lightPos + cfg.LightSize * RandHemisphere(rand, lightToRay);
    *lightDir = normalize(movedLightPos - rayPos);
    if (Reaches(rayPos, movedLightPos))
    {
        float div = length2(rayPos - movedLightPos);
        return LightBrightness(hue) / div;
    }
    return 0.0f;
}

__device__ static float BRDF(float3 normal, float3 incoming, float3 outgoing)
{
    float3 halfV = normalize(incoming + outgoing);
    float angle = acos(dot(normal, halfV));
    return 1 + Gauss(cfg.SpecularHighlightAmount, 0, cfg.SpecularHighlightSize, angle);
}


__device__ static float RenderingEquation(float3 rayPos, float3 rayDir, float qualityMul, float hue, ulong* rand)
{
    float total = 0;
    float color = 1;
    int isFog;
    int i = 0;
    do
    {
        float quality = i==0?cfg.QualityFirstRay*qualityMul:cfg.QualityRestRay;
        int hitLightsource;
        float distance = Trace(rayPos, rayDir, quality, hue, rand, &isFog, &hitLightsource);
        if (hitLightsource)
        {
            if (i == 0)
            {
                total += color * LightBrightness(hue);
            }
            break;
        }
        if (distance > cfg.MaxRayDist)
        {
            isFog = 1;
            break;
        }

        float3 newRayPos = rayPos + rayDir * distance;
        float3 newRayDir;

        float3 normal;
        if (isFog)
        {
            newRayDir = RandSphere(rand);
            //color *= FogColor(hue);
        }
        else
        {
            normal = Normal(newRayPos);
            newRayDir = RandHemisphere(rand, normal);
            color *= DeColor(newRayPos, hue) * cfg.Reflectivity;
        }

        float3 lightingRayDir;
        float direct = DirectLighting(newRayPos, hue, rand, &lightingRayDir);

        if (!isFog)
        {
            color *= BRDF(normal, newRayDir, -rayDir) * dot(normal, newRayDir);
            direct *= BRDF(normal, lightingRayDir, -rayDir) * dot(normal, lightingRayDir);
        }
        total += color * direct;

        rayPos = newRayPos;
        rayDir = newRayDir;

        if (isFog)
        {
            break;
        }
    } while (++i < cfg.NumRayBounces);
    if (isFog)
    {
        total += color * AmbientBrightness(hue);
    }
    return total;
}

__device__ static float3 HueToRGB(float hue, float value)
{
    hue *= 4;
    float frac = fmod(hue, 1.0f);
    float3 color;
    switch ((int)hue)
    {
        case 0:
            color = make_float3(frac, 0, 0);
            break;
        case 1:
            color = make_float3(1 - frac, frac, 0);
            break;
        case 2:
            color = make_float3(0, 1 - frac, frac);
            break;
        case 3:
            color = make_float3(0, 0, 1 - frac);
            break;
        default:
            color = make_float3(value);
            break;
    }
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    color *= value;
    return color;
}

__device__ static uint PackPixel(float4 pixel)
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

extern "C" __global__ void kern(
        uint* __restrict__ screenPixels,
        int screenX, int screenY, int width, int height,
        int frame)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float3 pos = make_float3(Camera.posX, Camera.posY, Camera.posZ);
    float3 look = make_float3(Camera.lookX, Camera.lookY, Camera.lookZ);
    float3 up = make_float3(Camera.upX, Camera.upY, Camera.upZ);

    ulong rand;
    uint2 randBuffer = BufferRand[y * width + x];
    rand = (ulong)randBuffer.x << 32 | (ulong)randBuffer.y;
    rand += y * width + x;
    if (frame < 1)
    {
        for (int i = 0; (float)i / cfg.RandSeedInitSteps - 1 < Rand(&rand) * cfg.RandSeedInitSteps; i++)
        {
        }
    }

    float hue = Rand(&rand);

    float2 screenCoords = make_float2((float)(x + screenX), (float)(y + screenY));
    screenCoords += make_float2(Rand(&rand) - 0.5f, Rand(&rand) - 0.5f);
    float fov = Camera.fov * 2 / (width + height);
    fov *= exp((hue - 0.5f) * cfg.FovAbberation);
    float3 rayDir = RayDir(look, up, screenCoords, fov);
    ApplyDof(&pos, &rayDir, Camera.focalDistance, hue, &rand);
    int screenIndex = y * width + x;

    float4 final;
    if (frame < 0)
    {
        float dist = SimpleTrace(pos, rayDir, 1 / fov);
        dist = sqrt(dist);
        BufferScratch[screenIndex] = final = make_float4(dist, dist, dist, 100);
    }
    else
    {
        const float weight = 1;
        float intensity = RenderingEquation(pos, rayDir, 1 / fov, hue, &rand);
        if (!cfg.BrightThresh)
        {
            intensity = fmin(intensity, cfg.BrightThresh);
        }
        float3 color = HueToRGB(hue, intensity) + make_float3(cfg.ColorBiasR, cfg.ColorBiasG, cfg.ColorBiasB);

        float4 old = BufferScratch[screenIndex];
        float3 oldxyz = make_float3(old.x, old.y, old.z);
        float3 diff = oldxyz - color;
        if (!isnan(color.x) && !isnan(color.y) && !isnan(color.z) && !isnan(weight))
        {
            if (frame != 0 && old.w + weight > 0)
                final = make_float4((color * weight + oldxyz * old.w) / (old.w + weight), old.w + weight);
            else
                final = make_float4(color, weight);
        }
        else
        {
            if (frame != 0)
                final = old;
            else
                final = make_float4(0);
        }
        BufferScratch[screenIndex] = final;
    }
    screenPixels[screenIndex] = PackPixel(final);
    BufferRand[screenIndex] = make_uint2((uint)(rand >> 32), (uint)rand);
}
