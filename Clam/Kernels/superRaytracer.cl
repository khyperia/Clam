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

void AddPixel(__global float4* screen, int width, int height, float3 color, int screenX, int screenY, int frame)
{
	if (screenX < 0 || screenY < 0 || screenX >= width || screenY >= height)
		return;

	float3 old = screen[screenY * width + screenX].xyz;
	if (isnan(old.x))
		old.x = 0.0f;
	if (isnan(old.y))
		old.y = 0.0f;
	if (isnan(old.z))
		old.z = 0.0f;
	screen[screenY * width + screenX].xyz = old + color;
}

void DivPixel(__global float4* screen, int width, int height, int screenX, int screenY, int frame)
{
	if (screenX < 0 || screenY < 0 || screenX >= width || screenY >= height)
		return;

	screen[screenY * width + screenX].xyz = (float3)(0,0,0);
}

float3 TracePath(float3 rayPos, float3 rayDir, float3 lightPos, unsigned int* rand)
{
	float3 emittanceResults[NumRayBounces];
	float3 bdrfResults[NumRayBounces];
	int i;
	for (i = 0; i < NumRayBounces; i++)
	{
		float distanceToNearest = Trace(rayPos, rayDir, 256, rand);
		if (distanceToNearest > MaxRayDist)
		{
			break;
		}

		rayPos = rayPos + rayDir * distanceToNearest;
		float3 normal = Normal(rayPos);
		do { rayDir = RandVec(rand);
		} while (dot(rayDir, normal) < 0);

		float reflectance = 3;
		
		float3 emittance;
		float rayPosDotLightPos = dot(normalize(rayPos), normalize(lightPos));
		if (rayPosDotLightPos > 0 && Reaches(rayPos, lightPos))
			emittance = rayPosDotLightPos * 0.5 * (float3)(EmittanceColor);
		else
			emittance = 0;

		emittanceResults[i] = emittance;

		float3 BDRF = 2 * reflectance * dot(rayDir, normal) * (float3)(BdrfColor);

		bdrfResults[i] = BDRF;
	}
	float3 final = (float3)(0);
	for (int reverseI = i - 1; reverseI >= 0; reverseI--)
		final = emittanceResults[reverseI] + bdrfResults[reverseI] * final;
	return final;
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

	unsigned int rand = get_global_id(0) * width * height + get_global_id(1) * width + frame * 10;
	for (int i = 0; i < 4; i++)
		Rand(&rand);

	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);
	float3 rayDir = RayDir(look, up, screenCoords, fov);

	float3 lightPosition = pos + (look + cross(look, up)) * (focalDistance / 4);

	float3 color = TracePath(pos, rayDir, lightPosition, &rand);
	
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
	screen[screenIndex] = (float4)(color, 1.0);
}
