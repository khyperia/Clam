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

float DeColor(float3 z3d, float lightHue)
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
    float fullValue = lightHue - hue * HueVariance;
    fullValue *= 3.14159;
    fullValue = cos(fullValue);
    fullValue *= fullValue;
    fullValue = pow(fullValue, ColorSharpness);
    fullValue = 1 - (1 - fullValue) * Saturation;
    return fullValue;
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

// Box-Muller transform
// returns two normally-distributed independent variables
float2 RandNormal(ulong* rand)
{
    float mul = sqrt(-2 * log(Rand(rand)));
    float angle = 6.28318530718 * Rand(rand);
    return mul * (float2)(cos(angle), sin(angle));
}

void ApplyDof(float3* position, float3* lookat, float focalPlane, float hue, ulong* rand)
{
    //focalPlane *= exp((hue - 0.5) * ChromaticAbberation);
    float3 focalPosition = *position + *lookat * focalPlane;
    float3 xShift = cross((float3)(0, 0, 1), *lookat);
    float3 yShift = cross(*lookat, xShift);
    float2 offset = RandCircle(rand);
    float dofPickup = DofAmount(hue);
    *lookat = normalize(*lookat + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
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

float Trace(float3 origin, float3 direction, float quality, float hue, ulong* rand,
        int* isFog, int* hitLightsource)
{
    float distance = 1.0;
    float totalDistance = 0.0;
    float sphereDist = RaySphereIntersection(origin, direction, (float3)(LightPos), LightSize);
    float fogDist = -log(Rand(rand)) / (float)(FogDensity(hue));
    fogDist = *isFog == 2 ? FLT_MAX : fogDist;
    float maxRayDist = min((float)MaxRayDist, min(fogDist, sphereDist));
    for (int i = 0; i < MaxRaySteps && totalDistance < maxRayDist &&
            distance * quality > totalDistance; i++) {
        distance = De(origin + direction * totalDistance) * DeMultiplier;
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
    else
        *isFog = 0;
    return totalDistance;
}

float SimpleTrace(float3 origin, float3 direction, float quality)
{
    float distance = 1.0;
    float totalDistance = 0.0;
    float sphereDist = RaySphereIntersection(origin, direction, (float3)(LightPos), LightSize);
    float maxRayDist = min((float)MaxRayDist, sphereDist);
    int i;
    for (i = 0; i < MaxRaySteps && totalDistance < maxRayDist &&
            distance * quality > totalDistance; i++)
    {
        distance = De(origin + direction * totalDistance) * DeMultiplier;
        totalDistance += distance;
    }
    return (float)i / MaxRaySteps;
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

float BRDF(float3 normal, float3 incoming, float3 outgoing)
{
    float3 halfV = normalize(incoming + outgoing);
    float angle = acos(dot(normal, halfV));
    return 1 + SpecularHighlight(angle);
}

float WeakeningFactor(float3 normal, float3 incoming)
{
    return dot(normal, incoming);
}

float DirectLighting(float3 newRayPos, float hue, ulong* rand, float3* lightDir)
{
    float3 light = normalize((float3)(RandNormal(rand), RandNormal(rand).x));
    *lightDir = normalize(light * LightSize + (float3)(LightPos) - newRayPos);
    if (dot(light, *lightDir) > 0)
        *lightDir = normalize(-light * LightSize + (float3)(LightPos) - newRayPos);
    int isFogLight = 2, isHitLight;
    float dist = Trace(newRayPos, *lightDir, QualityRestRay, hue, rand, &isFogLight, &isHitLight);

    if (isHitLight)
        return LightBrightness(hue) / (dist * dist); // clearly wrong, but I don't care
    return 0;
}

float RenderingEquation(float3 rayPos, float3 rayDir, float qualityMul, float hue, ulong* rand)
{
    // not entirely too sure why I need the 1 constant here
    // that is, "why not another number? This one is arbitrary" (pun sort of intended)
    float color = 1;
    float total = 0;
    int isFog = 0;
    for (int i = 0; i < NumRayBounces; i++)
    {
        int hitLightsource;

        float distanceTraveled = Trace(rayPos, rayDir,
                i==0?QualityFirstRay*qualityMul:QualityRestRay,
                hue, rand, &isFog, &hitLightsource);

        if (hitLightsource && i == 0)
        {
            total += color * LightBrightness(hue);
            break;
        }
        if (distanceTraveled >= MaxRayDist)
        {
            isFog = 1; // TODO: Is this required?
            break;
        }

        float3 newRayPos = rayPos + rayDir * distanceTraveled;
        float3 newRayDir;

        float3 normal;
        if (isFog)
        {
            newRayDir = normalize((float3)(RandNormal(rand), RandNormal(rand).x));
            color *= FogColor(hue);
        }
        else
        {
            normal = Normal(newRayPos);
            newRayDir = Cone(normal, 3.14 / 2, rand);
            color *= DeColor(newRayPos, hue);
        }

        float3 lightingRayDir;
        float direct = DirectLighting(newRayPos, hue, rand, &lightingRayDir);

        if (!isFog)
        {
            color *= BRDF(normal, newRayDir, -rayDir) *
                WeakeningFactor(normal, newRayDir);
            direct *= BRDF(normal, lightingRayDir, -rayDir)
                * WeakeningFactor(normal, lightingRayDir);
        }
        total += color * direct;

        rayPos = newRayPos;
        rayDir = newRayDir;
    }
    if (isFog)
        total += color * AmbientBrightness(hue);
    return total;
}

float3 HueToRGB(float hue, float value)
{
    hue *= 4;
    float frac = fmod(hue, 1);
    float3 color;
    switch ((int)hue)
    {
        case 0:
            color = (float3)(frac, 0, 0);
            break;
        case 1:
            color = (float3)(1 - frac, frac, 0);
            break;
        case 2:
            color = (float3)(0, 1 - frac, frac);
            break;
        case 3:
            color = (float3)(0, 0, 1 - frac);
            break;
        default:
            color = (float3)(value);
            break;
    }
    color = sqrt(color);
    color *= value;
    return color;
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

    float hue = Rand(&rand);

    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    fov *= exp((hue - 0.5) * FovAbberation);
    float3 rayDir = RayDir(look, up, screenCoords, fov);
    ApplyDof(&pos, &rayDir, focalDistance, hue, &rand);
    int screenIndex = y * width + x;

    if (frame == 0)
    {
        float dist = SimpleTrace(pos, rayDir, 1 / fov);
        dist = sqrt(dist);
        screen[screenIndex] = (float4)(dist);
    }
    else
    {
        frame -= 1;

        const float weight = 1;
        float intensity = RenderingEquation(pos, rayDir, 1 / fov, hue, &rand);
        float3 color = HueToRGB(hue, intensity);
        float4 final;

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
    }
    rngBuffer[screenIndex] = (uint2)((uint)(rand >> 32), (uint)rand);
}
