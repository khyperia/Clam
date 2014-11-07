#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdio.h>

#define CUBE_SIZE 256

#define SPHERE 0.5

inline int abs(int val)
{
    return val < 0 ? -val : val;
}

int colornormal(int ix, int iy, int iz, float* result)
{
    double x0 = (double)ix / CUBE_SIZE - 0.5;
    double y0 = (double)iy / CUBE_SIZE - 0.5;
    double z0 = (double)iz / CUBE_SIZE - 0.5;
    double len0 = sqrt(x0 * x0 + y0 * y0 + z0 * z0);

    double extralen = (len0 - SPHERE / 2) * CUBE_SIZE;

    if (fabs(extralen) > sqrt(3))
        return 0;

    double x = (ix + 0.5) / CUBE_SIZE * 2 - 1;
    double y = (iy + 0.5) / CUBE_SIZE * 2 - 1;
    double z = (iz + 0.5) / CUBE_SIZE * 2 - 1;
    double len = sqrt(x * x + y * y + z * z);

    x /= len;
    y /= len;
    z /= len;

    result[0] = (float)fabs(x);
    result[1] = (float)fabs(y);
    result[2] = (float)fabs(z);
    result[3] = (float)x;
    result[4] = (float)y;
    result[5] = (float)z;
    result[6] = (float)extralen;
    return 1;
}

long recurse(int x, int y, int z, int s, FILE* file)
{
    if (s == 1)
    {
        float result[7];
        if (colornormal(x, y, z, result))
        {
            long pos = ftell(file);
            fwrite(result, sizeof(float), sizeof(result) / sizeof(float), file);
            return pos;
        }
        return -1;
    }

    int s2 = s / 2;
    int i = 0;

    long leaves[8];

    for (int dx = 0; dx < 2; dx++)
    {
        for (int dy = 0; dy < 2; dy++)
        {
            for (int dz = 0; dz < 2; dz++)
            {
                long value = recurse(x + dx * s2, y + dy * s2, z + dz * s2, s2, file);
                leaves[i++] = value;
            }
        }
    }
    long curpos = ftell(file);
    int isWrite = 0;
    int toWrite[8];
    for (i = 0; i < 8; i++)
    {
        if (leaves[i] == -1)
        {
            toWrite[i] = 0;
        }
        else
        {
            toWrite[i] = (int)(leaves[i] - curpos); // will probably end up negative
            isWrite = 1;
        }
    }
    if (!isWrite)
        return -1;
    fwrite(toWrite, sizeof(int), 8, file);
    return curpos;
}

int run_makevoxel(lua_State* luaState)
{
    const char* filename = luaL_checkstring(luaState, 1);
    remove(filename);
    FILE* file = fopen(filename, "wb");
    int zero = 0;
    fwrite(&zero, sizeof(int), 1, file);
    int header = (int)recurse(0, 0, 0, CUBE_SIZE, file);
    fseek(file, 0, SEEK_SET);
    fwrite(&header, sizeof(int), 1, file);
    fclose(file);
    return 0;
}

int makevoxel(lua_State* luaState)
{
    lua_register(luaState, "makevoxel", run_makevoxel);
    return 0;
}
