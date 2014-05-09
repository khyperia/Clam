#ifndef MaxIters
#define MaxIters 64
#endif

#ifndef Bailout
#define Bailout 128
#endif

#ifndef Scale
#define Scale -1.5
#endif

#ifndef FoldingLimit
#define FoldingLimit 1.0
#endif

#ifndef FixedRadius2
#define FixedRadius2 1.0
#endif

#ifndef MinRadius2
#define MinRadius2 0.125
#endif

#ifndef Saturation
#define Saturation 0.2
#endif

#ifndef HueVariance
#define HueVariance 2
#endif

float De(float3 z)
{
#ifndef JuliaCenter
	float3 offset = z;
#else
	float3 offset = (float3)(JuliaCenter);
#endif
	float dz = 1.0;
	for (int n = 0; n < MaxIters; n++) {
		z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0 - z;

		float r2 = dot(z, z);

		if (r2 > Bailout)
			break;

		if (r2 < MinRadius2) { 
			float temp = FixedRadius2 / MinRadius2;
			z *= temp;
			dz *= temp;
		} else if (r2 < FixedRadius2) { 
			float temp = FixedRadius2 / r2;
			z *= temp;
			dz *= temp;
		}

		z = Scale * z + offset;
		dz = dz * fabs(Scale) + 1.0;
	}
	return length(z) / dz;
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

float3 DeColor(float3 z)
{
#if !defined(JuliaCenter)
	float3 offset = z;
#else
	float3 offset = (float3)(JuliaCenter);
#endif
	float hue = 0.0;
	for (int n = 0; n < MaxIters; n++) {
		float3 zOld = z;
		z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0 - z;
		zOld -= z;
		if (dot(zOld, zOld) < 0.01)
			hue += 7;

		float r2 = dot(z, z);

		if (r2 > Bailout)
			break;

		if (r2 < MinRadius2) { 
			float temp = FixedRadius2 / MinRadius2;
			z *= temp;
			hue += 11;
		} else if (r2 < FixedRadius2) { 
			float temp = FixedRadius2 / r2;
			z *= temp;
			hue += 13;
		}

		z = Scale * z + offset;
	}
	return HSVtoRGB(hue * HueVariance, Saturation, 1.0);
}
