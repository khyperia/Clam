#define dx(x, y, z) 10 * (y - x)
#define dy(x, y, z) x * (28 - z) - y
#define dz(x, y, z) x * y - 8.0/3 * z
#define Step 0.0004

#define ItersPerKernel 100

#define Brightness 0.5
#define MultiThreaded 1

float3 updatePos(float3 pos)
{
    float3 deriv = (float3)(
            dx(pos.x, pos.y, pos.z),
            dy(pos.x, pos.y, pos.z),
            dz(pos.x, pos.y, pos.z)
            );
    return pos + deriv * Step;
}

uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / UINT_MAX;
}

float3 genRand(int x, int y)
{
    ulong rand = (long)x << 32 | y;
    for (int i = 0; i < 64; i++)
        Rand(&rand);
    float3 result = (float3)(Rand(&rand), Rand(&rand), Rand(&rand));
    result = result * 2 - 1;
    result *= 100;
    return result;
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
    if (h < 0)
        h += 360;
    h /= 60;            // sector 0 to 5
    i = floor( h );
    f = h - i;          // factorial part of h
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
        default:        // case 5:
            return (float3)(v,p,q);
    }
}

float2 inverseCamera(float3 pos, float theta, float phi)
{
    float3 look = (float3)(cos(theta)*sin(phi),sin(theta)*sin(phi),cos(phi));
    const float3 cup = (float3)(0, 0, 1);
    float3 right = normalize(cross(look, cup));
    float3 up = normalize(cross(look, right));
    float3 invx = (float3)(right.x, up.x, look.x);
    float3 invy = (float3)(right.y, up.y, look.y);
    float3 invz = (float3)(right.z, up.z, look.z);

    float3 proj = pos.x * invx + pos.y * invy + pos.z * invz;
    return proj.xy * 0.02;
}

__kernel void main(__global float4* screen,
        int sx, int sy, int width, int height,
        __global float4* posbuffer, int frame,
        float theta, float phi)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    if (frame == 0)
        screen[y * width + x] = (float4)(0);

    if (MultiThreaded || (x == 0 && y == 0))
    {
        float3 pos = frame == 0 ?
            genRand(x + sx, y + sy) :
            posbuffer[y * width + x].xyz;

        for (int i = 0; i < ItersPerKernel; i++)
        {
            pos = updatePos(pos);
            float2 screenpos = inverseCamera(pos - (float3)(0, 0, 26), theta, phi);
            int2 screencoords = convert_int2(screenpos * (float2)(width, height)
                    - (float2)(sx, sy));
            if (screencoords.x >= 0 && screencoords.y >= 0 &&
                    screencoords.x < width && screencoords.y < height)
            {
                float value = MultiThreaded ?
                    Brightness / (ItersPerKernel * (float)(frame + 100))
                    : Brightness;
                screen[screencoords.y * width + screencoords.x].xyz += HSVtoRGB(
                        log((float)frame + 1) * 360, 0.5, value);
            }
        }
        posbuffer[y * width + x].xyz = pos;
    }
}
