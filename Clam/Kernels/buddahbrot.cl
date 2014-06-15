#ifndef Bailout
#define Bailout 4
#endif

#ifndef MaxIters
#define MaxIters 1<<10
#endif

#ifndef Gamma
#define Gamma 1
#endif

#ifndef PixelBrightness
#define PixelBrightness 1
#endif

uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

flt Rand(ulong* seed)
{
	return (flt)MWC64X(seed) / UINT_MAX;
}

float3 HSVtoRGB(float h, float s, float v)
{
	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		return (float3)(v);
	}
	h = fmod(h, 360);
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			return (float3)(v,t,p);
			break;
		case 1:
			return (float3)(q,v,p);
		case 2:
			return (float3)(p,v,t);
		case 3:
			return (float3)(p,q,v);
		case 4:
			return (float3)(t,p,v);
		default:		// case 5:
			return (float3)(v,p,q);
	}
}

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, flt xCenter, flt yCenter, flt zoom, int frame)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x < width && y < height)
	{
		int screenIndex = screenWidth ? (y - get_global_offset(1)) * screenWidth + (x - get_global_offset(0)) : y * width + x;
		if (frame == 0)
			screen[screenIndex] = (float4)(0);
		else
			screen[screenIndex].xyz *= (float)frame / (frame + 1);
	}

	if (get_group_id(0) > 5 || get_group_id(1) > 5)
		return;

	ulong rand = (ulong)get_global_id(0) + (ulong)get_global_id(1) * get_global_size(0) + (ulong)frame * get_global_size(0) * get_global_size(1);
	for (int i = 0; i < 4; i++)
		MWC64X(&rand);
	
	flt2 c = (flt2)(Rand(&rand) * 4 - 2, Rand(&rand) * 4 - 2);
	flt2 z = c;
	int2 hits[MaxIters];
	int it;
	for (it = 0; it < MaxIters; it++)
    {
		flt x2 = z.x * z.x;
		flt y2 = z.y * z.y;
		if (x2 + y2 > Bailout * Bailout)
			break;
		z = (flt2)(x2 - y2, 2 * z.x * z.y) + c;
	
		int2 screenCoords = convert_int2_rtn((float2)(((z.x - xCenter) / zoom + 1) / 2 * width, ((z.y * width / height - yCenter) / zoom + 1) / 2 * height));
		hits[it] = screenCoords;
    }
	if (it != MaxIters)
	{
		for (int thing = 0; thing < it; thing++)
		{
			int2 screenCoords = hits[thing];
			if (screenCoords.x >= 0 && screenCoords.x < width && screenCoords.y >= 0 && screenCoords.y < height)
			{
				int screenIndex = screenWidth ? (screenCoords.y - get_global_offset(1)) * screenWidth + (screenCoords.x - get_global_offset(0)) : screenCoords.y * width + screenCoords.x;
				float3 old = pow(screen[screenIndex].xyz, (float)Gamma);
				old += (float3)(PixelBrightness) * HSVtoRGB(log(thing + 1.0f) / log((float)(MaxIters)) * 360, 0.5, 1) / (frame + 1);
				screen[screenIndex].xyz = pow(old, 1 / (float)Gamma);
			}
		}
	}
}
