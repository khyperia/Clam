#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

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
