#include <../samples/common/inc/helper_math.h>
#include <float.h>
#include "kernelStructs.h"
#include "mandelbox.h"

#define Gauss(a, c, w, x) ((a) * exp(-(((x) - (c)) * ((x) - (c))) / (float)(2 * (w) * (w))))

__constant__ MandelboxCfg MandelboxCfgArr[1];
#define cfg (MandelboxCfgArr[0])

__constant__ GpuCameraSettings CameraArr[1];
#define Camera (CameraArr[0])

__constant__ float4* BufferScratchArr[1];
#define BufferScratch (BufferScratchArr[0])

__constant__ uint2* BufferRandArr[1];
#define BufferRand (BufferRandArr[0])

// http://en.wikipedia.org/wiki/Stereographic_projection
__device__ float3 RayDir(float3 forward, float3 up, float2 screenCoords, float fov) {
    screenCoords *= -fov;
    float len2 = dot(screenCoords, screenCoords);
    float3 look = make_float3(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1)
            / -(len2 + 1);

    float3 right = cross(forward, up);

    return look.x * right + look.y * up + look.z * forward;
}

__device__ float3 Rotate(float3 a, float angle)
{
    return make_float3(
        cos(angle) * a.x - sin(angle) * a.y,
        sin(angle) * a.x + cos(angle) * a.y,
        a.z
    );
}

__device__ float De(float3 offset) {
    offset = Rotate(offset, cfg.InitRotation);
    float4 z = make_float4(offset, 1.0f);
    const float FoldingLimit = cfg.FoldingLimit;
    const float MinRadius2 = cfg.MinRadius2;
    const float FixedRadius2 = cfg.FixedRadius2;
    const float Scale = cfg.Scale;
    const float DeRotation = cfg.DeRotation;
    const int MaxIters = cfg.MaxIters;
    const float Bailout = cfg.Bailout;
    for (int n = 0; n < MaxIters; n++) {
        float3 znew = make_float3(z.x, z.y, z.z);
        znew = clamp(znew, -FoldingLimit, FoldingLimit) * 2.0f - znew;
        z = make_float4(znew.x, znew.y, znew.z, z.w);

        float len2 = dot(znew, znew);
        if (len2 > Bailout)
            break;
        z *= FixedRadius2 / clamp(len2, MinRadius2, FixedRadius2);

        z = make_float4(Scale, Scale, Scale, fabs(Scale)) * z
                + make_float4(offset.x, offset.y, offset.z, 1.0f);

        float3 zRot = make_float3(z.x, z.y, z.z);
        zRot = Rotate(zRot, DeRotation);
        z = make_float4(zRot.x, zRot.y, zRot.z, z.w);
    }
    float3 zxyz = make_float3(z.x, z.y, z.z);
    return length(zxyz) / z.w;
}

__device__ float DeColor(float3 offset, float lightHue) {
    offset = Rotate(offset, cfg.InitRotation);
    float3 z = offset;
    int hue = 0;
    const float FoldingLimit = cfg.FoldingLimit;
    const float MinRadius2 = cfg.MinRadius2;
    const float FixedRadius2 = cfg.FixedRadius2;
    const float Scale = cfg.Scale;
    const float DeRotation = cfg.DeRotation;
    const int MaxIters = cfg.MaxIters;
    const float Bailout = cfg.Bailout;
    for (int n = 0; n < MaxIters && dot(z, z) < Bailout; n++) {
        float3 zold = z;
        z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0f - z;
        if (dot(zold - z, zold - z) > 0.0001f)
            hue += 7;
        else
            hue += 1;

        float len2 = dot(z, z);
        if (len2 > Bailout)
            break;
        float temp = FixedRadius2 / clamp(len2, MinRadius2, FixedRadius2);
        z *= temp;
        if (len2 < MinRadius2)
            hue += 0;
        else if (len2 < FixedRadius2)
            hue += 2;
        else
            hue += 5;

        z = Scale * z + offset;

        z = Rotate(z, DeRotation);
    }
    float fullValue = lightHue - hue * cfg.HueVariance;
    fullValue *= 3.14159f;
    fullValue = cos(fullValue);
    fullValue *= fullValue;
    fullValue = pow(fullValue, cfg.ColorSharpness);
    fullValue = 1 - (1 - fullValue) * cfg.Saturation;
    return fullValue;
}

/*
__device__ float De(float3 pos) {
    float3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    const float Power = cfg.Scale;
    const int MaxIters = cfg.MaxIters;
    const float Bailout = cfg.Bailout;
    for (int i = 0; i < MaxIters; i++) {
        r = length(z);
        if (r > Bailout) break;

        // convert to polar coordinates
        float theta = asin(z.z / r);
        float phi = atan2(z.y, z.x);
        dr = powf(r, Power - 1.0) * Power * dr + 1.0;

        // scale and rotate the point
        float zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;

        // convert back to cartesian coordinates
        z = zr * make_float3(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
        z += pos;
    }
    float distance = 0.5f * log(r) * r / dr;
    //distance = fabs(distance);
    if (pos.y > 0 && pos.z > 0)
    {
        float minPos = min(pos.y, pos.z);
        distance = max(distance, minPos);
    }
    return distance;
}

__device__ float DeColor(float3 pos, float lightHue) {
    float3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    const float Power = cfg.Scale;
    const int MaxIters = cfg.MaxIters;
    const float Bailout = cfg.Bailout;
    float hue = 0;
    for (int i = 0; i < MaxIters; i++) {
        r = length(z);
        if (r > Bailout) break;

        // convert to polar coordinates
        float theta = asin(z.z / r);
        float phi = atan2(z.y, z.x);
        dr = powf(r, Power - 1.0) * Power * dr + 1.0;

        hue += theta + phi;

        // scale and rotate the point
        float zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;

        // convert back to cartesian coordinates
        z = zr * make_float3(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
        z += pos;
    }
    float distance = 0.5f * log(r) * r / dr;
    if (pos.y > 0 && pos.z > 0 && min(pos.y, pos.z) > distance)
    {
        hue = length(z) / cfg.HueVariance;
    }
    float fullValue = lightHue - hue * cfg.HueVariance;
    fullValue *= 3.14159f;
    fullValue = cos(fullValue);
    fullValue *= fullValue;
    fullValue = pow(fullValue, cfg.ColorSharpness);
    fullValue = 1 - (1 - fullValue) * cfg.Saturation;
    return fullValue;
}
*/

__device__ uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

__device__ float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / UINT_MAX;
}

__device__ float2 RandCircle(ulong* rand)
{
    float2 polar = make_float2(Rand(rand) * 6.28318531f, sqrt(Rand(rand)));
    return make_float2(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

// Box-Muller transform
// returns two normally-distributed independent variables
__device__ float2 RandNormal(ulong* rand)
{
    float mul = sqrt(-2 * log2(Rand(rand)));
    float angle = 6.28318530718f * Rand(rand);
    return mul * make_float2(cos(angle), sin(angle));
}

__device__ float3 RandSphere(ulong* rand)
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

__device__ float3 RandHemisphere(ulong* rand, float3 normal)
{
    float3 result = RandSphere(rand);
    if (dot(result, normal) < 0)
        result = -result;
    return result;
}

__device__ void ApplyDof(float3* position, float3* lookat, float focalPlane, float hue, ulong* rand)
{
    float3 focalPosition = *position + *lookat * focalPlane;
    float3 xShift = cross(make_float3(0, 0, 1), *lookat);
    float3 yShift = cross(*lookat, xShift);
    float2 offset = RandCircle(rand);
    float dofPickup = cfg.DofAmount;
    *lookat = normalize(*lookat + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    *position = focalPosition - *lookat * focalPlane;
}

__device__ float3 Normal(float3 pos) {
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

__device__ float RaySphereIntersection(float3 rayOrigin, float3 rayDir,
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

__device__ float Trace(float3 origin, float3 direction, float quality, float hue, ulong* rand,
        int* isFog, int* hitLightsource)
{
    float distance = 1.0f;
    float totalDistance = De(origin) * cfg.DeMultiplier * Rand(rand);
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float sphereDist = RaySphereIntersection(origin, direction, lightPos, cfg.LightSize, false);
    float fogDist = -log2(Rand(rand)) / (float)(cfg.FogDensity /* * hue */);
    float maxRayDist = min(min((float)cfg.MaxRayDist, fogDist), sphereDist);
    for (int i = 0; i < cfg.MaxRaySteps && totalDistance < maxRayDist &&
            distance * quality > totalDistance; i++) {
        distance = De(origin + direction * totalDistance) * cfg.DeMultiplier;
        totalDistance += distance;
    }
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

__device__ float SimpleTrace(float3 origin, float3 direction, float quality)
{
    float distance = 1.0f;
    float totalDistance = 0.0f;
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float sphereDist = RaySphereIntersection(origin, direction, lightPos, cfg.LightSize, false);
    float maxRayDist = min((float)cfg.MaxRayDist, sphereDist);
    int i;
    for (i = 0; i < cfg.MaxRaySteps && totalDistance < maxRayDist &&
            distance * quality > totalDistance; i++)
    {
        distance = De(origin + direction * totalDistance) * cfg.DeMultiplier;
        totalDistance += distance;
    }
    return (float)i / cfg.MaxRaySteps;
}

__device__ bool Reaches(float3 initial, float3 final)
{
    float3 direction = final - initial;
    float lenDir = length(direction);
    direction /= lenDir;
    float totalDistance = 0;
    float distance = FLT_MAX;
    float threshHold = fabs(De(final)) * (cfg.DeMultiplier * 0.5f);
    for (int i = 0; i < cfg.MaxRaySteps && totalDistance < cfg.MaxRayDist &&
                distance > threshHold; i++) {
        distance = De(initial + direction * totalDistance) * cfg.DeMultiplier;
        if (i == 0 && fabs(distance * 0.5f) < threshHold)
            threshHold = fabs(distance * 0.5f);
        totalDistance += distance;
        if (totalDistance > lenDir)
            return true;
    }
    return false;
}

__device__ float LightBrightness(float hue)
{
    return Gauss(cfg.LightBrightnessAmount, cfg.LightBrightnessCenter, cfg.LightBrightnessWidth, hue);
}

__device__ float AmbientBrightness(float hue)
{
    return Gauss(cfg.AmbientBrightnessAmount, cfg.AmbientBrightnessCenter, cfg.AmbientBrightnessWidth, hue);
}

__device__ float DirectLighting(float3 rayPos, float hue, ulong* rand, float3* lightDir)
{
    const float3 lightPos = make_float3(cfg.LightPosX, cfg.LightPosY, cfg.LightPosZ);
    float3 lightToRay = normalize(rayPos - lightPos);
    float3 movedLightPos = lightPos + cfg.LightSize * RandHemisphere(rand, lightToRay);
    *lightDir = normalize(movedLightPos - rayPos);
    if (Reaches(rayPos, movedLightPos))
    {
        float div = dot(rayPos - movedLightPos, rayPos - movedLightPos);
        return LightBrightness(hue) / div;
    }
    return 0.0f;
}

__device__ float BRDF(float3 normal, float3 incoming, float3 outgoing)
{
    float3 halfV = normalize(incoming + outgoing);
    float angle = acos(dot(normal, halfV));
    return 1 + Gauss(cfg.SpecularHighlightAmount, 0, cfg.SpecularHighlightSize, angle);
}


__device__ float RenderingEquation(float3 rayPos, float3 rayDir, float qualityMul, float hue, ulong* rand)
{
    float total = 0;
    float color = 1;
    int isFog;
    for (int i = 0; i < cfg.NumRayBounces; i++)
    {
        bool isQuality = i == 0;
        float quality = isQuality?cfg.QualityFirstRay*qualityMul:cfg.QualityRestRay;
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
    }
    if (isFog)
    {
        total += color * AmbientBrightness(hue);
    }
    return total;
}

__device__ float3 HueToRGB(float hue, float value)
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

__device__ uint PackPixel(float4 pixel)
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
        BufferScratch[screenIndex] = final = make_float4(dist);
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
