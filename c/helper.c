#include "helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int PrintErrImpl(int errcode, const char* message, const char* filename, int line)
{
    if (errcode != 0)
        printf("%s(%d) `%s`: errcode %d\n", filename, line, message, errcode);
    return errcode;
}

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
    int len = strlen(str) + 1;
    char* ret = (char*)malloc(len * sizeof(char));
    if (!ret)
    {
        puts("malloc() failed");
        exit(-1);
    }
    for (int i = 0; i < len; i++)
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
    long fsize = ftell(f);
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
