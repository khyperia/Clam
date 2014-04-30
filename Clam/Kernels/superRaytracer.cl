#ifndef DeMultiplier
#define DeMultiplier 0.95
#endif

#ifndef DofPickup
#define DofPickup 0.005
#endif

#ifndef MaxRayDist
#define MaxRayDist 64
#endif

#ifndef MaxRaySteps
#define MaxRaySteps 256
#endif

#ifndef NumRayBounces
#define NumRayBounces 3
#endif

#ifndef QualityFirstRay
#define QualityFirstRay 1024
#endif

#ifndef QualityRestRay
#define QualityRestRay 64
#endif

#ifndef SpecularSize
#define SpecularSize 0.4
#endif

#ifndef SpecularDiffuseRatio
#define SpecularDiffuseRatio 0.5
#endif

#ifndef LightSize
#define LightSize 1.0
#endif

#ifndef LightBrightness
#define LightBrightness 2.0,1.5,1.0
#endif

#ifndef AmbientBrightness
#define AmbientBrightness 0.3,0.275,0.25
#endif

#ifndef LightReflectance
#define LightReflectance 1.0,1.5,2.0
#endif

float Rand(unsigned int* seed)
{
	const unsigned int scale = UINT_MAX / 32;
	return (float)((*seed = *seed * 1664525 + 1013904223) & scale) / scale;
}

// circular
bool IsGoodBokeh(float2 coords)
{
	float len2 = dot(coords, coords);
	return len2 < 1;
}

void ApplyDof(float3* position, float3* lookat, float focalPlane, unsigned int* rand)
{
	float3 focalPosition = *position + *lookat * focalPlane;
	float3 xShift = cross((float3)(0, 0, 1), *lookat);
	float3 yShift = cross(*lookat, xShift);
	float2 offset = (float2)(Rand(rand) * 6.28318531, Rand(rand));
	offset = (float2)(cos(offset.x) * offset.y, sin(offset.x) * offset.y);
	*lookat = normalize(*lookat + offset.x * DofPickup * xShift + offset.y * DofPickup * yShift);
	*position = focalPosition - *lookat * focalPlane;
}

float3 Normal(float3 pos) {
	const float delta = FLT_EPSILON * 2;
	float dppn = De(pos + (float3)(delta, delta, -delta));
	float dpnp = De(pos + (float3)(delta, -delta, delta));
	float dnpp = De(pos + (float3)(-delta, delta, delta));
	float dnnn = De(pos + (float3)(-delta, -delta, -delta));

	return normalize((float3)(
		(dppn + dpnp) - (dnpp + dnnn),
		(dppn + dnpp) - (dpnp + dnnn),
		(dpnp + dnpp) - (dppn + dnnn)
	));
}

float Trace(float3 origin, float3 direction, float quality, unsigned int* rand)
{
	float distance = De(origin) * Rand(rand) * DeMultiplier;
	float totalDistance = distance;
	for (int i = 0; i < MaxRaySteps && totalDistance < MaxRayDist && distance * quality > totalDistance; i++) {
		distance = De(origin + direction * totalDistance) * DeMultiplier;
		totalDistance += distance;
	}
	return totalDistance;
}

float3 Cone(float3 normal, float fov, unsigned int* rand)
{
	float2 coords = (float2)(Rand(rand) * 6.28318531, Rand(rand));
	coords = (float2)(cos(coords.x) * coords.y, sin(coords.x) * coords.y);
	float3 up = cross(cross(normal, (float3)(0,1,0)), normal);
	return RayDir(normal, up, coords, fov);
}

float3 TracePath(float3 rayPos, float3 rayDir, unsigned int* rand)
{
	float3 accum = (float3)(1);
	for (int i = 0; i < NumRayBounces; i++)
	{
		float distanceToNearest = Trace(rayPos, rayDir, i == 0 ? QualityFirstRay : QualityRestRay, rand);

		rayPos = rayPos + rayDir * distanceToNearest;

		if (distanceToNearest > MaxRayDist)
		{
			if (dot(normalize(rayPos), normalize((float3)(1,1,1))) > cos((float)LightSize))
				return accum * (float3)(LightBrightness);
			return accum * (float3)(AmbientBrightness);
		}
		
		float3 normal = Normal(rayPos);
		int isSpecular = Rand(rand) > SpecularDiffuseRatio;
		rayDir = Cone(isSpecular ? -2 * dot(normal, rayDir) * normal + rayDir : normal,
			isSpecular ? SpecularSize : 1.5,
			rand);

		accum *= (float3)(LightReflectance) * dot(rayDir, normal);
	}
	return (float3)(0.0f);
}

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, float4 position, float4 lookat, float4 updir, float fov, float focalDistance, int frame)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;

	float3 pos = position.xyz;
	float3 look = lookat.xyz;
	float3 up = updir.xyz;

	unsigned int rand = get_global_id(0) * 13 + get_global_id(1) * get_global_size(0) * 11 + frame * get_global_size(0) * get_global_size(1) * 7;
	for (int i = 0; i < 8; i++)
		Rand(&rand);

	ApplyDof(&pos, &look, focalDistance, &rand);

	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);
	float3 rayDir = RayDir(look, up, screenCoords, fov);

	//float3 lightPosition = pos + (look + cross(look, up)) * (focalDistance / 4);

	float3 color = TracePath(pos, rayDir, &rand);
	
	int screenIndex = screenWidth ? (y - get_global_offset(1)) * screenWidth + (x - get_global_offset(0)) : y * width + x;
	if (frame > 0)
	{
		int frameFixed = frame - 1;
		float3 old = screen[screenIndex].xyz;
		if (!isnan(old.x) && !isnan(old.y) && !isnan(old.z))
			color = (color + old * frameFixed) / (frameFixed + 1);
		else if (isnan(color.x) || isnan(color.y) || isnan(color.z))
			color = old;
	}
	screen[screenIndex] = (float4)(clamp(color, 0.0, 1.0), 1.0);
}
