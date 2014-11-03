#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdio.h>

#define CUBE_SIZE 256

int colornormal(int ix, int iy, int iz, float* result)
{
    double x = (double)ix / CUBE_SIZE * 2 - 1;
    double y = (double)iy / CUBE_SIZE * 2 - 1;
    double z = (double)iz / CUBE_SIZE * 2 - 1;
    double len = sqrt(x * x + y * y + z * z);
    if (len > 0.5 || len < 0.4)
        return 0;
    //printf("%d %d %d\n", ix, iy, iz);
    if (len <= 0)
        len = 0.0001;

    x /= len;
    y /= len;
    z /= len;

    result[0] = 1;
    result[1] = 1;
    result[2] = 1;
    result[3] = (float)x;
    result[4] = (float)y;
    result[5] = (float)z;
    return 1;
}

long recurse(int x, int y, int z, int s, FILE* file)
{
    float result[6];
    if (s == 1)
    {
        if (colornormal(x, y, z, result))
        {
            long pos = ftell(file);
            fwrite(result, sizeof(float), 6, file);
            return pos;
        }
        return 0;
    }
    long curpos = ftell(file);
    int s2 = s / 2;
    int i = 0;
    for (int dx = 0; dx < 2; dx++)
    {
        for (int dy = 0; dy < 2; dy++)
        {
            for (int dz = 0; dz < 2; dz++)
            {
                long filepos = recurse(x + dx * s2, y + dy * s2, z + dz * s2, s2, file);
                if (filepos)
                    filepos -= curpos;
                long finalPos = ftell(file);
                fseek(file, curpos + i * (long)sizeof(long), SEEK_SET);
                fwrite(&filepos, sizeof(long), 1, file);
                fseek(file, finalPos, SEEK_SET);
                i++;
            }
        }
    }
    return curpos;
}

int run_makevoxel(lua_State* luaState)
{
    const char* filename = luaL_checkstring(luaState, 1);
    FILE* file = fopen(filename, "wb");
    recurse(0, 0, 0, CUBE_SIZE, file);
    fclose(file);
    return 0;
}

int makevoxel(lua_State* luaState)
{
    lua_register(luaState, "makevoxel", run_makevoxel);
    return 0;
}
