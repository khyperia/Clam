#define CUBE_SIZE 256
#define LOG2_CUBE_SIZE 8

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

typedef union
{
    struct
    {
        int offsets[8];
    } branch;
    struct
    {
        // DO NOT use a floatN here, as they must be alinged to an N*4 byte boundary
        // (which they aren't due to wacked-out storage format of the octree)
        float colornormal[6];
    } leaf;
} Octree;

// Finds 0-7 index of pos (which is in the range of 0-1), and returns it
// It also re-scales pos to be in the range of 0-1 of the octree section
int applyOctree(float3* pos)
{
    uint3 offset = abs(*pos > 0.5);
    *pos = *pos * 2 - convert_float3(offset);
    return offset.x * 4 + offset.y * 2 + offset.z;
}

// Does the reverse operation of applyOctree, un-scaling pos back to it's original quadrant
// 'index' parameter is from applyOctree
void undoOctree(float3* pos, int index)
{
    int3 offset = (int3)((index & 4) >> 2, (index & 2) >> 1, index & 1);
    *pos = (*pos + convert_float3(offset)) / 2.0;
}

float minnz(float left, float right)
{
    if (left < 0)
        return right;
    if (right < 0)
        return left;
    return min(left, right);
}

float minvec(float3 vec)
{
    return minnz(minnz(vec.x, vec.y), vec.z);
}

#define VOXEL_ADVANCE_DIST_EPS 0.001

float VoxelAdvanceDist(float3 pos, float3 look)
{
    // 0 = look * x + pos
    // -pos / look = x
    // (1 - pos) / look = x

    return minnz(minvec(-pos / look), minvec((1 - pos) / look)) + VOXEL_ADVANCE_DIST_EPS;
}

float RayPlaneIntersection(float3 pos, float3 look, float3 plane)
{
    return -(dot(pos, plane) + dot((float3)(0.5), plane)) / dot(look, plane);
}

// This function works. Why? I don't know. How? I don't know. It just does.
float3 Trace(float3 pos, float3 look, __global Octree* octree, float3* color, float3* normal)
{
    float3 originalPos = pos;
    int stack[LOG2_CUBE_SIZE];
    __global Octree* stackOct[LOG2_CUBE_SIZE];
    int stackDepth = 0;
    while (true)
    {
        while (true)
        {
            if (stackDepth == LOG2_CUBE_SIZE)
            {
                color->x = octree->leaf.colornormal[0];
                color->y = octree->leaf.colornormal[1];
                color->z = octree->leaf.colornormal[2];
                normal->x = octree->leaf.colornormal[3];
                normal->y = octree->leaf.colornormal[4];
                normal->z = octree->leaf.colornormal[5];

                /*
                float3 newPos = pos + look * VoxelAdvanceDist(pos, look);
                if (dot(newPos - (float3)(0.5), normal) > 0)
                {
                    pos = newPos;
                    break;
                }
                */

                pos -= look * (1.1 * VOXEL_ADVANCE_DIST_EPS);
                while (stackDepth >= 0)
                    undoOctree(&pos, stack[stackDepth--]);
                return pos;
            }
            if (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1)
            {
                return (float3)(-1);
            }
            stack[stackDepth] = applyOctree(&pos);
            int offset = octree->branch.offsets[stack[stackDepth]];
            if (offset == 0)
            {
                pos += look * VoxelAdvanceDist(pos, look);
                undoOctree(&pos, stack[stackDepth]);
                break;
            }
            stackOct[stackDepth] = octree;
            octree = (__global Octree*)((__global char*)octree + offset);
            stackDepth++;
        }

        while (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1)
        {
            stackDepth--;
            if (stackDepth == -1)
                return pos;
            undoOctree(&pos, stack[stackDepth]);
            octree = stackOct[stackDepth];
        }
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

float3 RenderingEquation(float3 rayPos, float3 rayDir, __global Octree* octree, ulong* rand)
{
    float3 colorAccum = (float3)(1);
    float3 total = (float3)(0);
    for (int i = 0; i < 2; i++)
    {
        float3 color, normal;
        rayPos = Trace(rayPos, rayDir, octree, &color, &normal);
        if (rayPos.x < 0 || rayPos.x >= 1 ||
            rayPos.y < 0 || rayPos.y >= 1 ||
            rayPos.z < 0 || rayPos.z >= 1)
        {
            if (rayPos.x > 0.5 && rayPos.y > 0.5 && rayPos.z > 0.5)
                total += colorAccum * (float3)(2);
            break;
        }
        colorAccum *= color;
        rayDir = Cone(normal, 3.0 / 2, rand);
        rayPos += 2.0 / CUBE_SIZE * rayDir;
    }
    return total;
}

__kernel void Main(__global float4* screen, __global Octree* octree,
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

    octree = (__global Octree*)((__global char*)octree + *(__global int*)octree);

    float3 pos = (float3)(posX, posY, posZ);
    float3 look = (float3)(lookX, lookY, lookZ);
    float3 up = (float3)(upX, upY, upZ);

    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    float3 rayDir = RayDir(look, up, screenCoords, fov);

    ulong rand = frame * width * height + y * width + x;
    for (int i = 0; i < 100; i++)
        Rand(&rand);

    float3 new = RenderingEquation(pos, rayDir, octree, &rand);
    float3 old = screen[y * width + x].xyz;
    if (frame != 0)
        new = (new + old * frame) / (frame + 1);
    screen[y * width + x].xyz = new;
}
