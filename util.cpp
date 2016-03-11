#include "util.h"

int CudaContext::currentContext = -1;
const CudaContext CudaContext::Invalid = CudaContext(NULL, -1);
