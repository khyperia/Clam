#ifndef DeMultiplier
#define DeMultiplier 0.95
#endif

#ifndef DofPickup
#define DofPickup 0.005
#endif

#ifndef MotionBlur
#define MotionBlur 0.0,0.0,0.0
#endif

#ifndef MaxRayDist
#define MaxRayDist 64
#endif

#ifndef MaxRaySteps
#define MaxRaySteps 256
#endif

#ifndef NumRayBounces
#define NumRayBounces 2
#endif

#ifndef QualityFirstRay
#define QualityFirstRay 1024
#endif

#ifndef QualityRestRay
#define QualityRestRay 32
#endif

#ifndef SpecularSize
#define SpecularSize 0.4
#endif

#ifndef SpecularDiffuseRatio
#define SpecularDiffuseRatio 0.6
#endif

#ifndef LightBrightness
#define LightBrightness 2.0,1.0,0.7
#endif

#ifndef AmbientBrightness
#define AmbientBrightness 0.3,0.3,0.6
#endif

#ifndef SurfaceColor
#define SurfaceColor 0.7,0.9,0.8
#endif

#ifndef LightPos
#define LightPos 5
#endif

#ifndef LightSize
#define LightSize 1
#endif

float Rand(unsigned int* seed)
{
	const unsigned int scale = UINT_MAX / 32;
	return (float)((*seed = *seed * 1664525 + 1013904223) & scale) / scale;
}

float2 RandCircle(unsigned int* rand)
{
	float2 polar = (float2)(Rand(rand) * 6.28318531, sqrt(Rand(rand)));
	return (float2)(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

void ApplyDof(float3* position, float3* lookat, float focalPlane, unsigned int* rand)
{
	float3 focalPosition = *position + *lookat * focalPlane;
	float3 xShift = cross((float3)(0, 0, 1), *lookat);
	float3 yShift = cross(*lookat, xShift);
	float2 offset = RandCircle(rand);
	*lookat = normalize(*lookat + offset.x * DofPickup * xShift + offset.y * DofPickup * yShift);
	*position = focalPosition - *lookat * focalPlane;
}

void ApplyMotionBlur(float3* position, float3 lookat, float3 up, float focalDistance, unsigned int* rand)
{
	float amount = Rand(rand) * 2 - 1;
	float3 right = cross(lookat, up);
	float3 motionBlur = (float3)(MotionBlur) * focalDistance * amount;
	*position += motionBlur.x * right + motionBlur.y * up + motionBlur.z * lookat;
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
	float distance = 1.0;
	float totalDistance = 0.0;
	for (int i = 0; i < MaxRaySteps && totalDistance < MaxRayDist && distance * quality > totalDistance; i++) {
		distance = De(origin + direction * totalDistance) * DeMultiplier;
		totalDistance += distance;
	}
	return totalDistance;
}

int Reaches(float3 source, float3 dest)
{
	float3 direction = dest - source;
	float len = length(direction);
	direction /= len;
	float totalDistance = 0.0;
	for (int i = 0; i < 64; i++)
	{
		float distance = De(source + direction * totalDistance);
		totalDistance += distance;
		if (distance * QualityRestRay < totalDistance)
			return 0;
		float3 dist = source + direction * totalDistance - dest;
		if (totalDistance > len)
			return 1;
	}
	return 0;
}

float3 Cone(float3 normal, float fov, unsigned int* rand)
{
	float3 up = cross(cross(normal, (float3)(0,1,0)), normal);
	return RayDir(normal, up, RandCircle(rand), fov);
}

float3 DirectLighting(float3 position, float3 lightPos)
{
	float thing = dot(normalize(lightPos - position), normalize(lightPos));
	return thing > cos(0.1) && Reaches(position, lightPos) ? (float3)(LightBrightness) : (float3)(0.0);
}

float3 TracePath(float3 rayPos, float3 rayDir, unsigned int* rand)
{
	float3 color = (float3)(1);
	float3 accum = (float3)(0);
	for (int i = 0; i < NumRayBounces; i++)
	{
		float distanceToNearest = Trace(rayPos, rayDir, i == 0 ? QualityFirstRay : QualityRestRay, rand);

		rayPos = rayPos + rayDir * distanceToNearest;

		if (distanceToNearest > MaxRayDist)
		{
			accum += (float3)(AmbientBrightness) * color;
			break;
		}
		
		float3 lightPos = (float3)(LightPos) + (float3)(Rand(rand) * 2 - 1, Rand(rand) * 2 - 1, Rand(rand) * 2 - 1) * (float)LightSize;
		float3 directLighting = DirectLighting(rayPos, lightPos);
		
		float3 normal = Normal(rayPos);
		int isSpecular = Rand(rand) > SpecularDiffuseRatio;
		if (isSpecular)
		{
			rayDir = Cone(-2 * dot(normal, rayDir) * normal + rayDir, SpecularSize, rand);
			if (dot(rayDir, normalize(lightPos - rayPos)) < cos(SpecularSize))
				directLighting = (float3)(0);
		}
		else
		{
			color *= 2 * dot(-rayDir, normal);
			float2 circ = RandCircle(rand);
			float3 right = cross((float3)(0, 1, 0), normal);
			float3 up = cross(right, (float3)(0, 1, 0));
			float forward = sqrt(1 - dot(circ, circ));
			rayDir = right * circ.x + up * circ.y + normal * forward;
			directLighting *= dot(normal, normalize(lightPos - rayPos));
		}

		color *= (float3)(SurfaceColor);
		accum += color * directLighting;
	}
	return accum;
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
	for (int i = 0; i < 7; i++)
		Rand(&rand);

	ApplyDof(&pos, &look, focalDistance, &rand);
	ApplyMotionBlur(&pos, look, up, focalDistance, &rand);

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
