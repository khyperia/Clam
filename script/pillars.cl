struct Pillars
{
    __global float4* data;
    int width;
    int height;
};

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

float Trace(float3 origin, float3 direction, float maxRayDist,
                   struct Pillars pillars, float3* normal, float3* color)
{
    const float backstepDelta = 0.001;
    if (direction.y < 0)
    {
        int2 gridPos = convert_int2_rtn(origin.xz);
        if (gridPos.x < 0 || gridPos.x >= pillars.width ||
                     gridPos.y < 0 || gridPos.y >= pillars.height)
            return 1000000000;
        float4 datapoint = pillars.data[gridPos.y * pillars.width + gridPos.x];
        float pillarHeight = datapoint.x + datapoint.y + datapoint.z;
        pillarHeight *= 256 / 3.0f;
        float intersectionDistance = (origin.y - pillarHeight) / -direction.y;
        int2 newGridPos = convert_int2_rtn(origin.xz + direction.xz * intersectionDistance);
        if (newGridPos.x == gridPos.x && newGridPos.y == gridPos.y)
        {
            *normal = (float3)(0, 1, 0);
            *color = datapoint.xyz;
            return intersectionDistance - backstepDelta;
        }
    }

    float2 dt = 1.0f / direction.xz;
    float2 t_next;
    if (direction.x > 0)
        t_next.x = (floor(origin.x) + 1 - origin.x) * dt.x;
    else
        t_next.x = (origin.x - floor(origin.x)) * dt.x;

    if (direction.z > 0)
        t_next.y = (floor(origin.z) + 1 - origin.z) * dt.y;
    else
        t_next.y = (origin.z - floor(origin.z)) * dt.y;
    t_next = fabs(t_next);
    dt = fabs(dt);

    for (int i = 0; i < MaxRaySteps; i++)
    {
        float distance = min(fabs(t_next.x), fabs(t_next.y));
        if (distance > maxRayDist)
            return distance;
        float3 finalPos = origin + direction * (distance + backstepDelta);
        int2 gridPos = convert_int2_rtn(finalPos.xz);
        if (gridPos.x < 0 || gridPos.y < 0 ||
            gridPos.x >= pillars.width || gridPos.y >= pillars.height)
            return 1000000000;
        float4 datapoint = pillars.data[gridPos.y * pillars.width + gridPos.x];
        float pillarHeight = datapoint.x + datapoint.y + datapoint.z;
        pillarHeight *= 256 / 3.0f;

        int stepx;
        if ((stepx = (fabs(t_next.x) < fabs(t_next.y))))
            t_next.x += dt.x;
        else
            t_next.y += dt.y;

        float intersectionDistance = (origin.y - pillarHeight) / -direction.y;
        if (direction.y > 0)
        {
            *color = datapoint.xyz;
            if (finalPos.y < pillarHeight)
            {
                if (stepx)
                    *normal = (float3)(-sign(direction.x), 0, 0);
                else
                    *normal = (float3)(0, 0, -sign(direction.z));
                return distance - backstepDelta;
            }
        }
        else if (intersectionDistance < min(fabs(t_next.x), fabs(t_next.y)))
        {
            *color = datapoint.xyz;
            if (intersectionDistance > distance)
            {
                *normal = (float3)(0, 1, 0);
            }
            else if (stepx)
            {
                intersectionDistance = distance;
                *normal = (float3)(-sign(direction.x), 0, 0);
            }
            else
            {
                intersectionDistance = distance;
                *normal = (float3)(0, 0, -sign(direction.z));
            }
            return intersectionDistance - backstepDelta;
        }
    }
    return 1000000000;
}

int Reaches(float3 source, float3 dest, struct Pillars pillars)
{
    float3 color, normal;
    float len = length(dest - source);
    return Trace(source, normalize(dest - source), len,
                         pillars, &normal, &color)
        > len;
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

float3 RenderingEquation(float3 rayPos, float3 rayDir, ulong* rand, struct Pillars pillars)
{
    // not entirely too sure why I need the 1 constant here
    // that is, "why not another number? This one is arbitrary" (pun sort of intended)
    float3 color = (float3)(1);
    float3 total = (float3)(0);
    for (int i = 0; i < NumRayBounces; i++)
    {
        float3 normal, matColor;
        float distanceTraveled = Trace(rayPos, rayDir, MaxRayDist, pillars, &normal, &matColor);

        if (distanceTraveled > MaxRayDist)
        {
            total += color * (float3)(AmbientBrightness);
            break;
        }

        float3 newRayPos = rayPos + rayDir * distanceTraveled;
        float3 newRayDir = Cone(normal, 3.1415926535 / 2, rand);

        color *= matColor;

        float3 lightPos = (float3)(LightPos) + (float3)(LightSize) *
            ((float3)(Rand(rand), Rand(rand), Rand(rand)) * 2 - 1);

        if (Reaches(newRayPos, lightPos, pillars))
        {
            // TODO: Not sure how to integrate direct light sampling
            // Currently sort of approximating as if the material itself is giving off light
            // (Le in the rendering equation)
            // This produces wonkyness in the BRDF function,
            // so I put DeColor before this because it seems logical (it should be in BRDF)
            float3 distToLightsource2Vec = newRayPos - lightPos;
            float distanceToLightsource2 = dot(distToLightsource2Vec, distToLightsource2Vec);
            distanceToLightsource2 += distanceTraveled * distanceTraveled * 0.05;
            float3 light = (float3)(LightBrightness) * LightBrightnessMultiplier;
            float bdrf = BRDF(newRayPos, normal, normalize(lightPos - newRayPos), -rayDir);
            float weak = WeakeningFactor(normal, -rayDir);
            total += weak / distanceToLightsource2 * bdrf * light * color;
        }

        color *= BRDF(newRayPos, normal, newRayDir, -rayDir) * WeakeningFactor(normal, newRayDir);

        rayPos = newRayPos;
        rayDir = newRayDir;
    }
    return total;
}

__kernel void Main(__global float4* screen,
    int screenX, int screenY, int width, int height,
    float posX, float posY, float posZ,
    float lookX, float lookY, float lookZ,
    float upX, float upY, float upZ,
    float fov, float focalDistance, float frame,
    __global float4* pillarsd, int pillarsX, int pillarsY)
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

    struct Pillars pillars;
    pillars.data = pillarsd;
    pillars.width = pillarsX;
    pillars.height = pillarsY;
    float3 color = RenderingEquation(pos, rayDir, &rand, pillars);

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
