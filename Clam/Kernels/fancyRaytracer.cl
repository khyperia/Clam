#ifndef NormDistCount
#define NormDistCount 5
#endif

#ifndef Quality
#define Quality 8
#endif

#ifndef AoStepcount
#define AoStepcount 5
#endif

#ifndef FogDensity
#define FogDensity 0.0625
#endif

#ifndef MaxRayDist
#define MaxRayDist 64
#endif

#ifndef DofPickup
#define DofPickup 0.005
#endif

#ifndef BackgroundColor
#define BackgroundColor 0.5,0.5,0.5
#endif

#ifndef LambertCosineColor
#define LambertCosineColor 0.2,0.2,0.25
#endif

#ifndef CameraLightColor
#define CameraLightColor 0.2,0.2,0.25
#endif

#ifndef AOColor
#define AOColor 0.25,0.0,0.25
#endif

#define IntMax 2147483647

int Rand(int seed) {
	return seed * 1664525 + 1013904223;
}

// circular
bool IsGoodBokeh(float2 coords)
{
	float len2 = dot(coords, coords);
	return len2 < 1;
}

// cardiotropic
//bool IsGoodBokeh(float2 coords)
//{
//	float len2 = dot(coords, coords);
//	return 0.5 < len2 && len2 < 1;
//}

// hearts! <3
//bool IsGoodBokeh(float2 coords)
//{
//	//float dist = fabs(coords.x) + fabs(coords.y);
//	return (coords.x < 0 || coords.y < 0);
//}

int ApplyDof(float3* position, float3* lookat, float focalPlane, int rand)
{
	float3 focalPosition = *position + *lookat * focalPlane;
	float3 xShift = cross((float3)(0, 0, 1), *lookat);
	float3 yShift = cross(*lookat, xShift);
	float2 offset;
	do
	{
		int randx = Rand(rand);
		int randy = Rand(randx);
		rand = randy;
		offset = (float2)((float)randx / IntMax, (float)randy / IntMax);
	} while (!IsGoodBokeh(offset));
	*lookat = normalize(*lookat + offset.x * DofPickup * xShift + offset.y * DofPickup * yShift);
	*position = focalPosition - *lookat * focalPlane;
	return rand;
}

float NormDist(float3 pos, float3 dir, float totalDistance, float quality) {
	for (int i = 0; i < NormDistCount; i++)
	{
		float distance = De(pos + dir * totalDistance);
		distance -= (totalDistance + distance) / quality;
		totalDistance += distance;
	}
	return totalDistance;
}

float Trace(float3 origin, float3 direction, float quality)
{
	float distance = De(origin);
	float totalDistance = distance;
	for (int i = 0; i < 512 && totalDistance < MaxRayDist && distance * quality > totalDistance; i++) {
		distance = De(origin + direction * totalDistance);
		distance -= (totalDistance + distance) / quality;
		totalDistance += distance;
	}
	totalDistance = NormDist(origin, direction, totalDistance, quality);
	return totalDistance;
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

float3 LambertCosine(float3 lightDir, float3 normal, float3 color) {
	float value = dot(-lightDir, normal) * 0.5 + 0.5;
	const float3 light = (float3)(1, 1, 1);
	const float3 dark = (float3)(color);
	return mix(dark, light, value);
}

float3 AO(float3 pos, float3 normal) {
	float delta = De(pos);
	float totalDistance = delta;
	for (int i = 0; i < AoStepcount; i++)
		totalDistance += De(pos + normal * totalDistance);

	float value = totalDistance / (delta * pown(2.0, AoStepcount));
	const float3 light = (float3)(1, 1, 1);
	const float3 dark = (float3)(AOColor);
	return mix(dark, light, value);
}

float3 Fog(float3 color, float focalDistance, float totalDistance) {
	float ratio = totalDistance / focalDistance;
	float value = 1 / exp(ratio * FogDensity);
	return mix((float3)(BackgroundColor), color, value);
}

float3 Postprocess(float totalDistance, float3 origin, float3 direction, float focalDistance)
{
	if (totalDistance > MaxRayDist)
		return (float3)(BackgroundColor);
	float3 finalPos = origin + direction * totalDistance;
	float3 normal = Normal(finalPos);
	float3 color = AO(finalPos, normal);
	color *= LambertCosine(direction, normal, (float3)(CameraLightColor));
	color *= LambertCosine((float3)(0.57735026919, 0.57735026919, 0.57735026919), normal, (float3)(LambertCosineColor));
	color = Fog(color, focalDistance, totalDistance);
	return clamp(color, 0.0, 1.0);
}

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, float4 position, float4 lookat, float4 updir, float fov, float focalDistance, int frame)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;

	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);

	float3 pos = position.xyz;
	float3 look = lookat.xyz;
	float3 up = updir.xyz;

	int rand = Rand(x * width * height + y * width + frame * 10);
	for (int i = 0; i < 4; i++)
		rand = Rand(rand);

	if (frame > 0)
		rand = ApplyDof(&pos, &look, focalDistance, rand);

	float3 rayDir = RayDir(look, up, screenCoords, fov);

	float totalDistance = Trace(pos, rayDir, sqrt((float)width) * Quality / fov);

	float3 color = Postprocess(totalDistance, pos, rayDir, focalDistance);
	
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
	screen[screenIndex] = (float4)(color, totalDistance);
}
