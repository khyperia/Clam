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

#ifndef EmittanceColor
#define EmittanceColor 1,0.5,0.75
#endif

#ifndef BdrfColor
#define BdrfColor 0.5,1,0.75
#endif

#ifndef QualityFirstRay
#define QualityFirstRay 512
#endif

#ifndef QualityRestRay
#define QualityRestRay 128
#endif

#ifndef LightSize
#define LightSize 1.0
#endif

#ifndef LightBrightness
#define LightBrightness 2.0,1.0,1.5
#endif

#ifndef AmbientBrightness
#define AmbientBrightness 0.25,0.25,0.3
#endif

#ifndef LightReflectance
#define LightReflectance 1.0,2.0,1.5
#endif

float Rand(unsigned int* seed)
{
	return (float)(*seed = *seed * 1664525 + 1013904223) / UINT_MAX;
}

float3 RandVec(unsigned int* seed)
{
	float3 result;
	do
	{
		result = (float3)(Rand(seed) * 2 - 1, Rand(seed) * 2 - 1, Rand(seed) * 2 - 1);
	} while (dot(result, result) > 1);
	return result;
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
	float2 offset;
	do
	{
		offset = (float2)(Rand(rand) * 2 - 1, Rand(rand) * 2 - 1);
	} while (!IsGoodBokeh(offset));
	*lookat = normalize(*lookat + offset.x * DofPickup * xShift + offset.y * DofPickup * yShift);
	*position = focalPosition - *lookat * focalPlane;
}

float3 Normal(float3 pos) {
	const float delta = FLT_EPSILON * 16;
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

int Reaches(float3 position, float3 destination)
{
	float initialDistance = De(position);
	for (int i = 0; i < MaxRaySteps * 2; i++) {
		float distance = De(position);
		float3 direction = destination - position;
		float directionLength = length(direction);
		if (distance > directionLength)
			return 1;
		if (distance < initialDistance * 0.5)
			break;
		position += direction / directionLength * distance;
	}
	return 0;
}

float3 Diffuse(float3 normal, float3 rayDir, unsigned int* rand)
{
	do { rayDir = normalize(RandVec(rand));
	} while (dot(rayDir, normal) < 0);
	return rayDir;
}

float3 Specular(float3 normal, float3 rayDir, unsigned int* rand)
{
	float3 target = -2 * dot(normal, rayDir) * normal + rayDir;
	float3 up = cross(cross(target, (float3)(0,1,0)), target);
	for (int i = 0; i < 8 && dot(rayDir, normal) < 0; i++)
		rayDir = RayDir(target, up, (float2)(Rand(rand) * 2 - 1, Rand(rand) * 2 - 1), 0.1);
	return rayDir;
}

float3 Mirror(float3 normal, float3 rayDir, unsigned int* rand)
{
	return -2 * dot(normal, rayDir) * normal + rayDir;
}

float3 TracePath(float3 rayPos, float3 rayDir, unsigned int* rand)
{
	float3 accum = (float3)(1);
	for (int i = 0; i < NumRayBounces; i++)
	{
		Rand(rand);
		float distanceToNearest = Trace(rayPos, rayDir, i == 0 ? QualityFirstRay : QualityRestRay, rand);

		rayPos = rayPos + rayDir * distanceToNearest;

		if (distanceToNearest > MaxRayDist)
		{
			if (dot(normalize(rayPos), normalize((float3)(1,1,1))) > cos((float)LightSize))
				return accum * (float3)(LightBrightness);
			return accum * (float3)(AmbientBrightness);
		}

		float3 normal = Normal(rayPos);
		rayDir = Specular(normal, rayDir, rand);

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

	unsigned int rand = get_global_id(0) + get_global_id(1) * width + frame * width * height;
	for (int i = 0; i < 5; i++)
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
