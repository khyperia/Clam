#ifndef UseDouble
#define UseDouble 0
#endif

#if UseDouble
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double flt;
typedef double2 flt2;
typedef double3 flt3;
#else
typedef float flt;
typedef float2 flt2;
typedef float3 flt3;
#endif
