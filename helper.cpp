#include "helper.h"
#include <stdexcept>

void HandleErrImpl(int errcode, const char* filename, int line)
{
    if (errcode != 0)
        throw std::runtime_error(std::string(filename).append("(").append(std::to_string(line))
                .append("): errcode ").append(std::to_string(errcode)));
}
