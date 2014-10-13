#include "helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* sgetenv(const char* name, const char* defaultValue)
{
    const char* result = getenv(name);
    if (!result)
    {
        printf("Warning: %s env var not set, defaulting to %s\n", name, defaultValue);
        result = defaultValue;
    }
    return result;
}

// because strdup is apparently not in <string.h>
char* my_strdup(const char* str)
{
    size_t len = strlen(str) + 1;
    char* ret = (char*)malloc(len * sizeof(char));
    if (!ret)
    {
        puts("malloc() failed");
        exit(-1);
    }
    for (size_t i = 0; i < len; i++)
        *(ret + i) = *(str + i);
    return ret;
}

char* readWholeFile(const char* filename)
{
    FILE* f = fopen(filename, "rb");
    if (!f)
    {
        perror("readWholeFile()");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    char* string = malloc(fsize + 1);
    if (!string)
    {
        puts("malloc() failed");
        exit(-1);
    }
    fread(string, fsize, 1, f);
    fclose(f);

    string[fsize] = 0;

    return string;
}
