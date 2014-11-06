#define CUBE_SIZE 256
#define LOG2_CUBE_SIZE 8

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

float VoxelAdvanceDist(float3 pos, float3 look)
{
    // 0 = look * x + pos
    // -pos / look = x
    // (1 - pos) / look = x

    return minnz(minvec(-pos / look), minvec((1 - pos) / look)) + 0.01;
}

// This function works. Why? I don't know. How? I don't know. It just does.
float3 Trace(float3 pos, float3 look, __global Octree* octree)
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
                float3 color, normal;
                color.x = octree->leaf.colornormal[0];
                color.y = octree->leaf.colornormal[1];
                color.z = octree->leaf.colornormal[2];
                normal.x = octree->leaf.colornormal[3];
                normal.y = octree->leaf.colornormal[4];
                normal.z = octree->leaf.colornormal[5];
                return fabs(normal);
            }
            if (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1)
            {
                return (float3)(1,0,0);
            }
            stack[stackDepth] = applyOctree(&pos);
            int offset = octree->branch.offsets[stack[stackDepth]];
            if (offset == 0)
            {
                break;
            }
            stackOct[stackDepth] = octree;
            octree = (__global Octree*)((__global char*)octree + offset);
            stackDepth++;
        }

        pos += look * VoxelAdvanceDist(pos, look);

        undoOctree(&pos, stack[stackDepth]);

        while (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1)
        {
            stackDepth--;
            if (stackDepth == -1)
                return (float3)(length(originalPos - pos));
            undoOctree(&pos, stack[stackDepth]);
            octree = stackOct[stackDepth];
        }
    }
}

float3 TraceConstDist(float3 pos, float3 look, __global Octree* octree)
{
    pos += look * 0.5;
    int stack[LOG2_CUBE_SIZE];
    int stackDepth = 0;
    while (true)
    {
        if (stackDepth == LOG2_CUBE_SIZE)
        {
            for (int i = LOG2_CUBE_SIZE - 1; i >= 0; i--)
            {
                undoOctree(&pos, stack[i]);
            }
            float x = octree->leaf.colornormal[3];
            float y = octree->leaf.colornormal[4];
            float z = octree->leaf.colornormal[5];
            return (float3)(x, y, z);
        }
        else
        {
            int index = applyOctree(&pos);
            int offset = octree->branch.offsets[index];
            if (offset == 0)
                return (float3)(1, 0, 0);
            octree = (__global Octree*)((__global char*)octree + offset);
            stack[stackDepth] = index;
            stackDepth++;
        }
    }
}

__kernel void Main(__global float4* screen, __global Octree* octree,
        int screenX, int screenY, int width, int height,
        float posX, float posY, float posZ,
        float lookX, float lookY, float lookZ,
        float upX, float upY, float upZ,
        float fov)
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

    screen[y * width + x].xyz = Trace(pos, rayDir, octree);
}
