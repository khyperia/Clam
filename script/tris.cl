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
    int numVerts;
    union
    {
        struct
        {
            int offset1;
            int verts[];
        } leaf;
        struct
        {
            float bbox[6];
            int offset1;
            int offset2;
        } branch;
    } u;
} Tri;

typedef struct
{
    __global Tri* tris;
    __global float4* verts;
    __global float4* normals;
    int offset;
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

float rayBbox(float3 o, float3 d, float3 minb, float3 maxb)
{
    float3 t1 = (minb - o) / d;
    float3 t2 = (maxb - o) / d;
    float3 tmin = min(t1, t2);
    float3 tmax = max(t1, t2);
    float dmin = max(max(tmin.x, tmin.y), tmin.z);
    float dmax = min(min(tmax.x, tmax.y), tmax.z);
    if (dmax > dmin && dmax > 0)
        return dmin;
    return FLT_MIN;
}

inline void grabTriData(float3 pos, float3 look, Tris tris, __global Tri* tri, int i,
        float* mini, float3* normal)
{
    int vert1i = tri->u.leaf.verts[i * 3 + 0];
    int vert2i = tri->u.leaf.verts[i * 3 + 1];
    int vert3i = tri->u.leaf.verts[i * 3 + 2];
    float3 vert1 = tris.verts[vert1i].xyz;
    float3 vert2 = tris.verts[vert2i].xyz;
    float3 vert3 = tris.verts[vert3i].xyz;
    float newMin = MollerTrumbore(vert1, vert2, vert3, pos, look);
    if (newMin < *mini)
    {
        *mini = newMin;
        float3 normal1 = tris.normals[vert1i].xyz;
        float3 normal2 = tris.normals[vert2i].xyz;
        float3 normal3 = tris.normals[vert3i].xyz;
        pos = pos + newMin * look;
        float n1w = length(pos - vert1);
        float n2w = length(pos - vert2);
        float n3w = length(pos - vert3);
        float sum = n1w + n2w + n3w;
        n1w = sum - n1w;
        n2w = sum - n2w;
        n3w = sum - n3w;
        *normal = normalize(normal1 * n1w + normal2 * n2w + normal3 * n3w);
    }
}

float Trace(float3 pos, float3 look, Tris tris, float3* color, float3* normal)
{
    float mini = FLT_MAX;
    *color = (float3)(0.5);

    while (true)
    {
        __global Tri* tri = (__global Tri*)((__global char*)tris.tris + tris.offset);
        int numVerts = tri->numVerts;
        if (numVerts == 0)
        {
            float3 minb =
                (float3)(tri->u.branch.bbox[0], tri->u.branch.bbox[1], tri->u.branch.bbox[2]);
            float3 maxb =
                (float3)(tri->u.branch.bbox[3], tri->u.branch.bbox[4], tri->u.branch.bbox[5]);
            float bbox = rayBbox(pos, look, minb, maxb);
            tris.offset = bbox == FLT_MIN || bbox > mini ?
                tri->u.branch.offset1 :
                tri->u.branch.offset2;
        }
        else
        {
            tris.offset = tri->u.leaf.offset1;
            for (int i = 0; i < numVerts; i++)
                grabTriData(pos, look, tris, tri, i, &mini, normal);
        }
        if (tris.offset == 0)
            return mini;
    }
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
                total += colorAccum * (float3)(5);
            break;
        }
        rayPos += rayDir * traceAmount;
        colorAccum *= color;
        rayDir = Cone(normal, 3.0 / 2, rand);
        rayPos += 0.001 * rayDir;
    }
    return total;
}

__kernel void Main(__global float4* screen, __global Tri* triArr,
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
    tris.verts = (__global float4*)((__global char*)triArr + *((__global int*)triArr + 1));
    tris.normals = (__global float4*)((__global char*)triArr + *((__global int*)triArr + 2));
    tris.offset = *(__global int*)triArr;

    float3 new = RenderingEquation(pos, rayDir, tris, &rand);
    float3 old = screen[y * width + x].xyz;
    if (frame != 0)
        new = (new + old * frame) / (frame + 1);
    screen[y * width + x].xyz = new;
}
