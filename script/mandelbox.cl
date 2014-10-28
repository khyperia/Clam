float3 Rotate(float3 u, float theta, float3 vec)
{
    float cost = cos(theta);
    float cost1 = 1 - cost;
    float sint = sin(theta);
    return (float3)(
            (cost + u.x * u.x * cost1) * vec.x +
            (u.x * u.y * cost1 - u.z * sint) * vec.y +
            (u.x * u.z * cost1 + u.y * sint) * vec.z,
            (u.y * u.x * cost1 + u.z * sint) * vec.x +
            (cost + u.y * u.y * cost1) * vec.y +
            (u.y * u.z * cost1 - u.x * sint) * vec.z,
            (u.z * u.x * cost1 - u.y * sint) * vec.x +
            (u.z * u.y * cost1 + u.x * sint) * vec.y +
            (cost + u.z * u.z * cost1) * vec.z
            );
}

float3 RayDir(float3 look, float3 up, float2 screenCoords, float fov)
{
    float angle = atan2(screenCoords.y, -screenCoords.x);
    float dist = length(screenCoords) * fov;

    float3 axis = Rotate(look, angle, up);
    float3 direction = Rotate(axis, dist, look);

    return direction;
}

float De(float3 z3d)
{
    // Note: Transform4D *must* be an orthogonal matrix, otherwise the De property doesn't hold
    float4 z = z3d.x * normalize((float4)(Transform4Dx)) +
        z3d.y * normalize((float4)(Transform4Dy)) +
        z3d.z * normalize((float4)(Transform4Dz));
    float4 offset = z;
    float dz = 1.0;
    for (int n = 0; n < MaxIters; n++) {
        z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0 - z;

        float r2 = dot(z, z);

        if (r2 > Bailout)
            break;

        if (r2 < MinRadius2) {
            float temp = FixedRadius2 / MinRadius2;
            z *= temp;
            dz *= temp;
        } else if (r2 < FixedRadius2) {
            float temp = FixedRadius2 / r2;
            z *= temp;
            dz *= temp;
        }

        z = Scale * z + offset;
        dz = dz * fabs(Scale) + 1.0;
    }
    return length(z) / dz;
}

float3 HSVtoRGB(float h, float s, float v)
{
    int i;
    float f, p, q, t;
    if( s == 0 ) {
        // achromatic (grey)
        return (float3)(v);
    }
    h = fmod(h, 360);
    h /= 60;            // sector 0 to 5
    i = floor( h );
    f = h - i;          // factorial part of h
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );
    switch( i ) {
        case 0:
            return (float3)(v,t,p);
            break;
        case 1:
            return (float3)(q,v,p);
        case 2:
            return (float3)(p,v,t);
        case 3:
            return (float3)(p,q,v);
        case 4:
            return (float3)(t,p,v);
        default:        // case 5:
            return (float3)(v,p,q);
    }
}

float3 DeColor(float3 z3d)
{
    float4 z = z3d.x * normalize((float4)(Transform4Dx)) +
        z3d.y * normalize((float4)(Transform4Dy)) +
        z3d.z * normalize((float4)(Transform4Dz));
    float4 offset = z;
    float hue = 0.0;
    for (int n = 0; n < MaxIters; n++) {
        float4 zOld = z;
        z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0 - z;
        zOld -= z;
        if (dot(zOld, zOld) < 0.01)
            hue += 7;

        float r2 = dot(z, z);

        if (r2 > Bailout)
            break;

        if (r2 < MinRadius2) {
            float temp = FixedRadius2 / MinRadius2;
            z *= temp;
            hue += 11;
        } else if (r2 < FixedRadius2) {
            float temp = FixedRadius2 / r2;
            z *= temp;
            hue += 13;
        }

        z = Scale * z + offset;
    }
    return HSVtoRGB(hue * HueVariance, Saturation, Reflectivity) + (float3)(ColorBias);
}

uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / UINT_MAX;
}

float2 RandCircle(ulong* rand)
{
    float2 polar = (float2)(Rand(rand) * 6.28318531, sqrt(Rand(rand)));
    return (float2)(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

void ApplyDof(float3* position, float3* lookat, float focalPlane, ulong* rand)
{
    float3 focalPosition = *position + *lookat * focalPlane;
    float3 xShift = cross((float3)(0, 0, 1), *lookat);
    float3 yShift = cross(*lookat, xShift);
    float2 offset = RandCircle(rand);
    *lookat = normalize(*lookat + offset.x * DofPickup * xShift + offset.y * DofPickup * yShift);
    *position = focalPosition - *lookat * focalPlane;
}

float3 Normal(float3 pos) {
    const float delta = FLT_EPSILON * 2;
    float dppn = De(pos + (float3)(delta, delta, -delta));
    float dpnp = De(pos + (float3)(delta, -delta, delta));
    float dnpp = De(pos + (float3)(-delta, delta, delta));
    float dnnn = De(pos + (float3)(-delta, -delta, -delta));

    return normalize((float3)(
                (dppn + dpnp) - (dnpp + dnnn),
                (dppn + dnpp) - (dpnp + dnnn),
                (dpnp + dnpp) - (dppn + dnnn)
                ));
}

float RaySphereIntersection(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereSize)
{
    float3 omC = rayOrigin - sphereCenter;
    float lDotOmC = dot(rayDir, omC);
    float underSqrt = lDotOmC * lDotOmC - dot(omC, omC) + sphereSize * sphereSize;
    if (underSqrt < 0)
        return FLT_MAX;
    float dist = -lDotOmC - sqrt(underSqrt);
    if (dist <= 0)
        return FLT_MAX;
    return dist;
}

float Trace(float3 origin, float3 direction, float quality, ulong* rand,
        int* isFog, int* hitLightsource)
{
    float distance = 1.0;
    float totalDistance = 0.0;
    float sphereDist = RaySphereIntersection(origin, direction, (float3)(LightPos), LightSize);
    float maxRayDist = min((float)MaxRayDist, sphereDist);
    *isFog = 0;
    for (int i = 0; i < MaxRaySteps && totalDistance < maxRayDist &&
            distance * quality > totalDistance; i++) {
        distance = De(origin + direction * totalDistance) * DeMultiplier;
        totalDistance += distance;

        // Fog. This code is magical mathematics.
        float density = exp(-(float)(FogDensity) * distance);
        float random = Rand(rand);
        if (random > density)
        {
            *isFog = 1;
            float backstep = -log(random) / (float)(FogDensity);
            totalDistance -= backstep;
            break;
        }
    }
    if (totalDistance > sphereDist)
        *hitLightsource = 1;
    else
        *hitLightsource = 0;
    return totalDistance;
}

float3 Cone(float3 normal, float fov, ulong* rand)
{
    const float3 dir1 = normalize((float3)(1,1,1));
    const float3 dir2 = normalize((float3)(-1,1,1));
    float3 dir = dir1;
    if (fabs(dot(normal, dir1)) > cos(0.2))
        dir = dir2;
    float3 up = cross(cross(normal, normalize(dir)), normal);
    return RayDir(normal, up, RandCircle(rand), fov);
}

float3 NewRayDir(float3 pos, float3 normal, float* weight, ulong* rand)
{
    float3 toLight = (float3)(LightPos) - pos;
    float toLightLen = length(toLight);
    toLight /= toLightLen;
    float angle = asin(LightSize / toLightLen);
    if (isnan(angle))
        angle = 1.58;
    float probLight = (LightSize * LightSize) / (toLightLen * toLightLen);

    if (Rand(rand) < DirectLightProbability)
    {
        *weight *= probLight / DirectLightProbability;
        float3 newRayDir = Cone(toLight, angle, rand);
        if (dot(newRayDir, normal) < 0)
            *weight = 0;
        return newRayDir;
    }
    else
    {
        *weight *= (1 - probLight) / (1 - DirectLightProbability);
        float3 newRayDir = Cone(normal, 3.14 / 2, rand);
        if (dot(newRayDir, toLight) > cos(angle))
            *weight = 0;
        return newRayDir;
    }
}

float3 NewSphereRayDir(float3 pos, float* weight, ulong* rand)
{
    float3 toLight = (float3)(LightPos) - pos;
    float toLightLen = length(toLight);
    toLight /= toLightLen;
    float angle = asin(LightSize / toLightLen);
    if (isnan(angle))
        angle = 2.23;
    float probLight = (LightSize * LightSize) / (toLightLen * toLightLen) / 2;

    if (Rand(rand) < DirectLightProbability)
    {
        *weight *= probLight / DirectLightProbability;
        return Cone(toLight, angle, rand);
    }
    else
    {
        *weight *= (1 - probLight) / (1 - DirectLightProbability);
        float3 newRayDir;
        do
        {
            newRayDir = (float3)(Rand(rand), Rand(rand), Rand(rand)) * 2 - 1;
        } while (dot(newRayDir, newRayDir) > 1);
        newRayDir = normalize(newRayDir);
        if (dot(newRayDir, toLight) > cos(angle))
            *weight = 0;
        return newRayDir;
    }
}

float BRDF(float3 point, float3 normal, float3 incoming, float3 outgoing)
{
    float3 halfV = normalize(incoming + outgoing);
    float angle = acos(dot(normal, halfV));
    float multiplier = 1 + SpecularBrightness *
        exp(-(angle * angle) / (SpecularSize * SpecularSize));
    return multiplier;
}

float WeakeningFactor(float3 normal, float3 incoming)
{
    return dot(normal, incoming);
}

float3 RenderingEquation(float3 rayPos, float3 rayDir, float qualityMul, float* weight, ulong* rand)
{
    // not entirely too sure why I need the 1 constant here
    // that is, "why not another number? This one is arbitrary" (pun sort of intended)
    float3 color = (float3)(1);
    float3 total = (float3)(0);
    int isFog;
    for (int i = 0; i < NumRayBounces; i++)
    {
        int hitLightsource;

        float distanceTraveled = Trace(rayPos, rayDir,
                i==0?QualityFirstRay*qualityMul:QualityRestRay, rand, &isFog, &hitLightsource);

        if (hitLightsource)
        {
            total += color * (float3)(LightBrightness) * LightBrightnessMultiplier;
            break;
        }
        if (distanceTraveled >= MaxRayDist)
        {
            isFog = 1;
            break;
        }

        float3 newRayPos = rayPos + rayDir * distanceTraveled;
        float3 newRayDir;

        if (isFog)
        {
            newRayDir = NewSphereRayDir(newRayPos, weight, rand);
            if (*weight == 0)
                return total; // doesn't matter anyway
            color *= (float3)(FogColor);
        }
        else
        {
            float3 normal = Normal(newRayPos);

            newRayDir = NewRayDir(newRayPos, normal, weight, rand);
            if (*weight == 0)
                return total; // doesn't matter anyway

            color *= DeColor(newRayPos) * BRDF(newRayPos, normal, newRayDir, -rayDir) *
                WeakeningFactor(normal, newRayDir);
        }
        rayPos = newRayPos;
        rayDir = newRayDir;
    }
    if (isFog)
        total += color * (float3)(AmbientBrightness) * AmbientBrightnessMultiplier;
    return total;
}

__kernel void Main(__global float4* screen,
        __global uint2* rngBuffer,
        int screenX, int screenY, int width, int height,
        float posX, float posY, float posZ,
        float lookX, float lookY, float lookZ,
        float upX, float upY, float upZ,
        float fov, float focalDistance, float frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    float3 pos = (float3)(posX, posY, posZ);
    float3 look = (float3)(lookX, lookY, lookZ);
    float3 up = (float3)(upX, upY, upZ);

    ulong rand;
    uint2 randBuffer = rngBuffer[y * width + x];
    rand = (ulong)randBuffer.x << 32 | (ulong)randBuffer.y;
    rand += (ulong)get_global_id(0) + (ulong)get_global_id(1) * get_global_size(0);
    if (frame == 0)
    {
        for (int i = 0; (float)i / RandSeedInitSteps - 1 < Rand(&rand) * RandSeedInitSteps; i++)
        {
        }
    }

    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    float3 rayDir = RayDir(look, up, screenCoords, fov);

    ApplyDof(&pos, &rayDir, focalDistance, &rand);

    float weight = 1;
    float3 color = RenderingEquation(pos, rayDir, 1 / fov, &weight, &rand);
    float4 final;

    int screenIndex = y * width + x;
    float4 old = screen[screenIndex];
    if (frame != 0 && old.w + weight > 0)
        final = (float4)((color * weight + old.xyz * old.w) / (old.w + weight), old.w + weight);
    else if (!isnan(color.x) && !isnan(color.y) && !isnan(color.z) && !isnan(weight))
        final = (float4)(color, weight);
    else if (frame != 0)
        final = old;
    else
        final = (float4)(0);
    screen[screenIndex] = final;
    rngBuffer[screenIndex] = (uint2)((uint)(rand >> 32), (uint)rand);
}
