#pragma once

#include <stdio.h>

#if __GNUC__
#define UNUSED __attribute__((__unused__))
#else
#define UNUSED
#endif

const char* sgetenv(const char* name, const char* defaultValue);
char* my_strdup(const char* str);
char* readWholeFile(const char* filename);
void* malloc_s(size_t size);
inline int PrintErrImpl(int errcode, const char* message, const char* filename, int line)
{
    if (errcode != 0)
        printf("%s(%d) `%s`: errcode %d\n", filename, line, message, errcode);
    return errcode;
}

#define PrintErr(errcode) PrintErrImpl((errcode), (#errcode), __FILE__, __LINE__)
