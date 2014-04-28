float3 Rotate(float3 u, float theta, float3 vec)
{
	float cost = cos(theta);
	float cost1 = 1 - cost;
	float sint = sin(theta);
	return (float3)(
		(cost + u.x * u.x * cost1) * vec.x +		(u.x * u.y * cost1 - u.z * sint) * vec.y +	(u.x * u.z * cost1 + u.y * sint) * vec.z,
		(u.y * u.x * cost1 + u.z * sint) * vec.x +	(cost + u.y * u.y * cost1) * vec.y +		(u.y * u.z * cost1 - u.x * sint) * vec.z,
		(u.z * u.x * cost1 - u.y * sint) * vec.x +	(u.z * u.y * cost1 + u.x * sint) * vec.y +	(cost + u.z * u.z * cost1) * vec.z
	);
}

float3 RayDir(float3 look, float3 up, float2 screenCoords, float fov)
{
	float angle = atan2(screenCoords.y, -screenCoords.x);
	float dist = length(screenCoords) * fov;

	float3 axis = Rotate(look, angle, up);
	float3 direction = Rotate(axis, dist, look);

	return direction;
}

float2 InverseRayDir(float3 look, float3 up, float3 direction, float fov)
{
	float3 right = cross(look, up);
	return (float2)(-dot(right, direction) / fov, -dot(up, direction) / fov);
}
