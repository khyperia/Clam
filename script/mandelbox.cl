#ifndef MaxIters
#define MaxIters 64
#endif

#ifndef Bailout
#define Bailout 128
#endif

#ifndef Scale
#define Scale -1.5
#endif

#ifndef FoldingLimit
#define FoldingLimit 1.0
#endif

#ifndef FixedRadius2
#define FixedRadius2 1.0
#endif

#ifndef MinRadius2
#define MinRadius2 0.125
#endif

#ifndef Saturation
#define Saturation 0.2
#endif

#ifndef HueVariance
#define HueVariance 2
#endif

#ifndef RandSeedInitSteps
#define RandSeedInitSteps 256
#endif

#ifndef DeMultiplier
#define DeMultiplier 0.95
#endif

#ifndef DofPickup
#define DofPickup 0.005
#endif

#ifndef MotionBlur
#define MotionBlur 0.0,0.0,0.0
#endif

#ifndef MaxRayDist
#define MaxRayDist 64
#endif

#ifndef MaxRaySteps
#define MaxRaySteps 256
#endif

#ifndef DirectLightingMaxSteps
#define DirectLightingMaxSteps 64
#endif

#ifndef NumRayBounces
#define NumRayBounces 2
#endif

#ifndef QualityFirstRay
#define QualityFirstRay 1024
#endif

#ifndef QualityRestRay
#define QualityRestRay 32
#endif

#ifndef SpecularSize
#define SpecularSize 0.4
#endif

#ifndef SpecularDiffuseRatio
#define SpecularDiffuseRatio 0.6
#endif

#ifndef LightBrightness
#define LightBrightness 1.5,1.4,1.0
#endif

#ifndef AmbientBrightness
#define AmbientBrightness 0.3,0.3,0.4
#endif

#ifndef LightPos
#define LightPos 5
#endif

#ifndef LightSize
#define LightSize 1
#endif

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

float De(float3 z)
{
#ifndef JuliaCenter
    float3 offset = z;
#else
    float3 offset = (float3)(JuliaCenter);
#endif
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

float3 DeColor(float3 z)
{
#if !defined(JuliaCenter)
    float3 offset = z;
#else
    float3 offset = (float3)(JuliaCenter);
#endif
    float hue = 0.0;
    for (int n = 0; n < MaxIters; n++) {
        float3 zOld = z;
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
    return HSVtoRGB(hue * HueVariance, Saturation, 1.0);
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

void ApplyMotionBlur(float3* position, float3 lookat, float3 up, float focalDistance, ulong* rand)
{
    float amount = Rand(rand) * 2 - 1;
    float3 right = cross(lookat, up);
    float3 motionBlur = (float3)(MotionBlur) * focalDistance * amount;
    *position += motionBlur.x * right + motionBlur.y * up + motionBlur.z * lookat;
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

float Trace(float3 origin, float3 direction, float quality, ulong* rand)
{
    float distance = 1.0;
    float totalDistance = 0.0;
    for (int i = 0; i < MaxRaySteps && totalDistance < MaxRayDist && distance * quality > totalDistance; i++) {
        distance = De(origin + direction * totalDistance) * DeMultiplier;
        totalDistance += distance;
    }
    return totalDistance;
}

int Reaches(float3 source, float3 dest)
{
    float3 direction = dest - source;
    float len = length(direction);
    direction /= len;
    float totalDistance = 0.0;
    for (int i = 0; i < DirectLightingMaxSteps; i++)
    {
        float distance = De(source + direction * totalDistance);
        totalDistance += distance;
        if (distance * QualityRestRay < totalDistance)
            return 0;
        float3 dist = source + direction * totalDistance - dest;
        if (totalDistance > len)
            return 1;
    }
    return 0;
}

float3 Cone(float3 normal, float fov, ulong* rand)
{
    float3 up = cross(cross(normal, (float3)(0,1,0)), normal);
    return RayDir(normal, up, RandCircle(rand), fov);
}

float3 DirectLighting(float3 position, float3 lightPos)
{
    //float thing = dot(normalize(lightPos - position), normalize(lightPos));
    return
        //thing > cos(0.1) &&
        Reaches(position, lightPos) ? (float3)(LightBrightness) : (float3)(0.0);
}

float3 TracePath(float3 rayPos, float3 rayDir, float focalDistance, ulong* rand)
{
    float3 color = (float3)(1);
    float3 accum = (float3)(0);
    for (int i = 0; i < NumRayBounces; i++)
    {
        float distanceToNearest = Trace(rayPos, rayDir, i == 0 ? QualityFirstRay : QualityRestRay, rand);

        rayPos = rayPos + rayDir * distanceToNearest;

        if (distanceToNearest > MaxRayDist)
        {
            accum += (float3)(AmbientBrightness) * color;
            break;
        }

        float3 lightPos = (float3)(LightPos) + (float3)(Rand(rand) * 2 - 1, Rand(rand) * 2 - 1, Rand(rand) * 2 - 1) * (float)LightSize;
        float3 directLighting = DirectLighting(rayPos, lightPos);
        
        float3 normal = Normal(rayPos);
        int isSpecular = Rand(rand) > SpecularDiffuseRatio;
        if (isSpecular)
        {
            rayDir = Cone(-2 * dot(normal, rayDir) * normal + rayDir, SpecularSize, rand);
            if (dot(rayDir, normalize(lightPos - rayPos)) < cos(SpecularSize))
                directLighting = (float3)(0);
        }
        else
        {
            color *= 2 * dot(-rayDir, normal);
            float2 circ = RandCircle(rand);
            float3 right = cross((float3)(0, 1, 0), normal);
            float3 up = cross(right, (float3)(0, 1, 0));
            float forward = sqrt(1 - dot(circ, circ));
            rayDir = right * circ.x + up * circ.y + normal * forward;
            directLighting *= dot(normal, normalize(lightPos - rayPos));
        }

        color *= DeColor(rayPos);
        accum += color * directLighting;
    }
    return accum;
}

__kernel void Main(__global float4* screen,
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

    ulong rand = (ulong)get_global_id(0) +
        (ulong)get_global_id(1) * get_global_size(0) +
        (ulong)frame * get_global_size(0) * get_global_size(1);
    for (int i = 0; i < RandSeedInitSteps; i++)
        MWC64X(&rand);

    ApplyDof(&pos, &look, focalDistance, &rand);
    ApplyMotionBlur(&pos, look, up, focalDistance, &rand);

    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    float3 rayDir = RayDir(look, up, screenCoords, fov);

    float3 color = TracePath(pos, rayDir, focalDistance, &rand);
    
    int screenIndex = y * width + x;
    if (frame > 0)
    {
        int frameFixed = (int)frame - 1;
        float3 old = screen[screenIndex].xyz;
        if (!isnan(old.x) && !isnan(old.y) && !isnan(old.z))
            color = (color + old * frameFixed) / (frameFixed + 1);
        else if (isnan(color.x) || isnan(color.y) || isnan(color.z))
            color = old;
    }
    screen[screenIndex] = (float4)(clamp(color, 0.0, 1.0), 1.0);
}
