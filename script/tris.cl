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

typedef struct
{
    float verts[9];
} Tri;

typedef struct
{
    __global Tri* tris;
    unsigned int size;
} Tris;

// http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
float MollerTrumbore(float3 v1, float3 v2, float3 v3, float3 o, float3 d)
{
    float3 e1 = v2 - v1;
    float3 e2 = v3 - v1;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (fabs(det) < FLT_EPSILON)
        return FLT_MAX;
    float inv_det = 1.0 / det;
    float3 t = o - v1;
    float u = dot(t, p) * inv_det;
    if (u < 0 || u > 1)
        return FLT_MAX;
    float3 q = cross(t, e1);
    float v = dot(d, q) * inv_det;
    if (v < 0 || u + v > 1)
        return FLT_MAX;
    float dist = dot(e2, q) * inv_det;
    if (dist < FLT_EPSILON)
        return FLT_MAX;
    return dist;
}

float Trace(float3 pos, float3 look, Tris tris, float3* color, float3* normal)
{
    float mini = FLT_MAX;
    *color = (float3)(0.5);
    for (unsigned int i = 0; i < tris.size; i++)
    {
        Tri tri = tris.tris[i];
        float3 vert1 = (float3)(tri.verts[0], tri.verts[1], tri.verts[2]);
        float3 vert2 = (float3)(tri.verts[3], tri.verts[4], tri.verts[5]);
        float3 vert3 = (float3)(tri.verts[6], tri.verts[7], tri.verts[8]);
        float newMin = MollerTrumbore(vert1, vert2, vert3, pos, look);
        if (newMin < mini)
        {
            mini = newMin;
            *normal = cross(normalize(vert2 - vert1), normalize(vert3 - vert1));
        }
    }
    return mini;
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

float3 RenderingEquation(float3 rayPos, float3 rayDir, Tris tris, ulong* rand)
{
    float3 colorAccum = (float3)(1);
    float3 total = (float3)(0);
    for (int i = 0; i < 2; i++)
    {
        float3 color, normal;
        float traceAmount = Trace(rayPos, rayDir, tris, &color, &normal);
        if (traceAmount >= FLT_MAX)
        {
            if (rayDir.x > 0.0 && rayDir.y > 0.0 && rayDir.z > 0.0)
                total += colorAccum * (float3)(1);
            break;
        }
        rayPos += rayDir * traceAmount;
        colorAccum *= color;
        rayDir = Cone(normal, 3.0 / 2, rand);
        rayPos += 0.001 * rayDir;
    }
    return total;
}

__kernel void Main(__global float4* screen, __global Tri* triArr, unsigned int triSize,
        int screenX, int screenY, int width, int height,
        float posX, float posY, float posZ,
        float lookX, float lookY, float lookZ,
        float upX, float upY, float upZ,
        float fov, int frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    float3 pos = (float3)(posX, posY, posZ);
    float3 look = (float3)(lookX, lookY, lookZ);
    float3 up = (float3)(upX, upY, upZ);

    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    float3 rayDir = RayDir(look, up, screenCoords, fov);

    ulong rand = frame * width * height + y * width + x;
    for (int i = 0; i < 100; i++)
        Rand(&rand);

    Tris tris;
    tris.tris = triArr;
    tris.size = triSize;

    float3 new = RenderingEquation(pos, rayDir, tris, &rand);
    float3 old = screen[y * width + x].xyz;
    if (frame != 0)
        new = (new + old * frame) / (frame + 1);
    screen[y * width + x].xyz = new;
}
