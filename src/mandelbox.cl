struct MandelboxCfg
{
    float _pos_x;                     // 0.0 1.0
    float _pos_y;                     // 0.0 1.0
    float _pos_z;                     // 5.0 1.0
    float _look_x;                    // 0.0 1.0
    float _look_y;                    // 0.0 1.0
    float _look_z;                    // -1.0 1.0
    float _up_x;                      // 0.0 1.0
    float _up_y;                      // 1.0 1.0
    float _up_z;                      // 0.0 1.0
    float _fov;                       // 1.0 -1.0
    float _focal_distance;            // 3.0 -1.0
    float _scale;                     // -2.0 0.5 const
    float _folding_limit;             // 1.0 -0.5 const
    float _fixed_radius_2;            // 1.0 -0.5 const
    float _min_radius_2;              // 0.125 -0.5 const
    float _dof_amount;                // 0.001 -0.5
    float _fog_distance;              // 10.0 -1.0
    float _fog_brightness;            // 1.0 0.5
    float _light_pos_1_x;             // 3.0 1.0
    float _light_pos_1_y;             // 3.5 1.0
    float _light_pos_1_z;             // 2.5 1.0
    float _light_radius_1;            // 1.0 -0.5
    float _light_brightness_1_hue;    // 0.0 0.25
    float _light_brightness_1_sat;    // 0.4 -1.0
    float _light_brightness_1_val;    // 4.0 -1.0
    float _ambient_brightness_hue;    // 0.65 0.25
    float _ambient_brightness_sat;    // 0.2 -1.0
    float _ambient_brightness_val;    // 1.0 -1.0
    float _reflect_brightness;        // 1.0 0.125
    float _surface_color_variance;    // 1.0 -0.25
    float _surface_color_shift;       // 0.0 0.25
    float _surface_color_saturation;  // 1.0 0.125
    float _bailout;                   // 1024.0 -1.0 const
    float _de_multiplier;             // 0.9375 0.125 const
    float _max_ray_dist;              // 16.0 -0.5 const
    float _quality_first_ray;         // 2.0 -0.5 const
    float _quality_rest_ray;          // 64.0 -0.5 const
    float _rotation;                  // 0.0 0.125
    int _white_clamp;                 // 0 const
    int _max_iters;                   // 64 const
    int _max_ray_steps;               // 256 const
    int _num_ray_bounces;             // 3 const
    int _preview;                     // 0 const
};

#ifndef pos_x
#define pos_x cfg->_pos_x
#endif
#ifndef pos_y
#define pos_y cfg->_pos_y
#endif
#ifndef pos_z
#define pos_z cfg->_pos_z
#endif
#ifndef look_x
#define look_x cfg->_look_x
#endif
#ifndef look_y
#define look_y cfg->_look_y
#endif
#ifndef look_z
#define look_z cfg->_look_z
#endif
#ifndef up_x
#define up_x cfg->_up_x
#endif
#ifndef up_y
#define up_y cfg->_up_y
#endif
#ifndef up_z
#define up_z cfg->_up_z
#endif
#ifndef fov
#define fov cfg->_fov
#endif
#ifndef focal_distance
#define focal_distance cfg->_focal_distance
#endif
#ifndef scale
#define scale cfg->_scale
#endif
#ifndef folding_limit
#define folding_limit cfg->_folding_limit
#endif
#ifndef fixed_radius_2
#define fixed_radius_2 cfg->_fixed_radius_2
#endif
#ifndef min_radius_2
#define min_radius_2 cfg->_min_radius_2
#endif
#ifndef dof_amount
#define dof_amount cfg->_dof_amount
#endif
#ifndef fog_distance
#define fog_distance cfg->_fog_distance
#endif
#ifndef fog_brightness
#define fog_brightness cfg->_fog_brightness
#endif
#ifndef light_pos_1_x
#define light_pos_1_x cfg->_light_pos_1_x
#endif
#ifndef light_pos_1_y
#define light_pos_1_y cfg->_light_pos_1_y
#endif
#ifndef light_pos_1_z
#define light_pos_1_z cfg->_light_pos_1_z
#endif
#ifndef light_radius_1
#define light_radius_1 cfg->_light_radius_1
#endif
#ifndef light_brightness_1_hue
#define light_brightness_1_hue cfg->_light_brightness_1_hue
#endif
#ifndef light_brightness_1_sat
#define light_brightness_1_sat cfg->_light_brightness_1_sat
#endif
#ifndef light_brightness_1_val
#define light_brightness_1_val cfg->_light_brightness_1_val
#endif
#ifndef ambient_brightness_hue
#define ambient_brightness_hue cfg->_ambient_brightness_hue
#endif
#ifndef ambient_brightness_sat
#define ambient_brightness_sat cfg->_ambient_brightness_sat
#endif
#ifndef ambient_brightness_val
#define ambient_brightness_val cfg->_ambient_brightness_val
#endif
#ifndef reflect_brightness
#define reflect_brightness cfg->_reflect_brightness
#endif
#ifndef surface_color_variance
#define surface_color_variance cfg->_surface_color_variance
#endif
#ifndef surface_color_shift
#define surface_color_shift cfg->_surface_color_shift
#endif
#ifndef surface_color_saturation
#define surface_color_saturation cfg->_surface_color_saturation
#endif
#ifndef bailout
#define bailout cfg->_bailout
#endif
#ifndef de_multiplier
#define de_multiplier cfg->_de_multiplier
#endif
#ifndef max_ray_dist
#define max_ray_dist cfg->_max_ray_dist
#endif
#ifndef quality_first_ray
#define quality_first_ray cfg->_quality_first_ray
#endif
#ifndef quality_rest_ray
#define quality_rest_ray cfg->_quality_rest_ray
#endif
#ifndef rotation
#define rotation cfg->_rotation
#endif
#ifndef white_clamp
#define white_clamp cfg->_white_clamp
#endif
#ifndef max_iters
#define max_iters cfg->_max_iters
#endif
#ifndef max_ray_steps
#define max_ray_steps cfg->_max_ray_steps
#endif
#ifndef num_ray_bounces
#define num_ray_bounces cfg->_num_ray_bounces
#endif
#ifndef preview
#define preview cfg->_preview
#endif

// Note: When num_ray_bounces is a dynamic variable in MandelboxCfg, the intel
// opencl runtime cannot vectorize the kernel.

typedef __private struct MandelboxCfg const* Cfg;

static float3 HueToRGB(float hue, float saturation, float value)
{
    hue *= 3;
    float frac = fmod(hue, 1.0f);
    float3 color;
    switch ((int)hue)
    {
        case 0:
            color = (float3)(1 - frac, frac, 0);
            break;
        case 1:
            color = (float3)(0, 1 - frac, frac);
            break;
        case 2:
            color = (float3)(frac, 0, 1 - frac);
            break;
        default:
            color = (float3)(1, 1, 1);
            break;
    }
    saturation = value * (1 - saturation);
    color = color * (value - saturation) + (float3)(saturation, saturation, saturation);
    return color;
}

static float3 LightPos1(Cfg cfg)
{
    return (float3)(light_pos_1_x, light_pos_1_y, light_pos_1_z);
}

static float3 LightBrightness1(Cfg cfg)
{
    return HueToRGB(light_brightness_1_hue, light_brightness_1_sat, light_brightness_1_val) /
           reflect_brightness;
}

static float3 AmbientBrightness(Cfg cfg)
{
    return HueToRGB(ambient_brightness_hue, ambient_brightness_sat, ambient_brightness_val) /
           reflect_brightness;
}

struct Random
{
    ulong seed;
};

static float Random_Next(struct Random* this)
{
    uint c = this->seed >> 32;
    uint x = this->seed & 0xFFFFFFFF;
    this->seed = x * ((ulong)4294883355U) + c;
    return (x ^ c) / 4294967296.0f;
}

static struct Random new_Random(uint idx, uint frame, uint global_size)
{
    struct Random result;
    result.seed = (ulong)idx + (ulong)global_size * (ulong)frame;
    for (int i = 0; i < 8; i++)
    {
        Random_Next(&result);
    }
    return result;
}

static float2 Random_Disk(struct Random* this)
{
    const float2 polar = (float2)(Random_Next(this) * 2.0f, sqrt(Random_Next(this)));
    return (float2)(cospi(polar.x) * polar.y, sinpi(polar.x) * polar.y);
}

static float3 Random_Sphere(struct Random* this)
{
    const float theta = Random_Next(this);
    const float phi = acospi(2 * Random_Next(this) - 1);
    const float x = sinpi(phi) * cospi(theta);
    const float y = sinpi(phi) * sinpi(theta);
    const float z = cospi(phi);
    return (float3)(x, y, z);
}

static float3 Random_Ball(struct Random* this)
{
    return rootn(Random_Next(this), 3) * Random_Sphere(this);
}

static float3 Random_Lambertian(struct Random* this, float3 normal)
{
    return normalize(Random_Ball(this) + normal);
}

struct Ray
{
    float3 pos;
    float3 dir;
};

static struct Ray new_Ray(float3 pos, float3 dir)
{
    struct Ray result;
    result.pos = pos;
    result.dir = dir;
    return result;
}

// http://en.wikipedia.org/wiki/Stereographic_projection
static float3 RayDir(float3 forward, float3 up, float2 screenCoords, float calcFov)
{
    screenCoords *= -calcFov;
    const float len2 = dot(screenCoords, screenCoords);
    const float3 look = (float3)(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);
    const float3 right = cross(forward, up);
    return look.x * right + look.y * up + look.z * forward;
}

static float3 Ray_At(struct Ray this, float time)
{
    return this.pos + this.dir * time;
}

static void Ray_Dof(Cfg cfg, struct Ray* this, float focalPlane, struct Random* rand)
{
    const float3 focalPosition = Ray_At(*this, focalPlane);
    const float3 xShift = cross((float3)(0, 0, 1), this->dir);
    const float3 yShift = cross(this->dir, xShift);
    const float2 offset = Random_Disk(rand);
    const float dofPickup = dof_amount;
    this->dir =
        normalize(this->dir + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    this->pos = focalPosition - this->dir * focalPlane;
}

static struct Ray Camera(Cfg cfg, uint x, uint y, uint width, uint height, struct Random* rand)
{
    const float3 origin = (float3)(pos_x, pos_y, pos_z);
    const float3 look = (float3)(look_x, look_y, look_z);
    const float3 up = (float3)(up_x, up_y, up_z);
    const float2 antialias = (float2)(Random_Next(rand), Random_Next(rand)) - (float2)(0.5f, 0.5f);
    const float2 screenCoords =
        (float2)((float)x - (float)(width / 2), (float)y - (float)(height / 2)) + antialias;
    const float calcFov = fov * 2 / (width + height);
    const float3 direction = RayDir(look, up, screenCoords, calcFov);
    struct Ray result = new_Ray(origin, direction);
    Ray_Dof(cfg, &result, focal_distance, rand);
    return result;
}

/*
static float4 Mandelbulb(float4 z, const float Power)
{
    const float r = length(z.xyz);
    // convert to polar coordinates
    float theta = asin(z.z / r);
    float phi = atan2(z.y, z.x);
    float dr = pow(r, Power - 1.0f) * Power * z.w + 1.0f;
    // scale and rotate the point
    float zr = pow(r, Power);
    theta = theta * Power;
    phi = phi * Power;
    // convert back to cartesian coordinates
    float3 z3 =
        zr * (float3)(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
    return (float4)(z3.x, z3.y, z3.z, dr);
}
*/

static float3 BoxfoldD(Cfg cfg, float3 z)
{
    return clamp(z, -folding_limit, folding_limit) * 2.0f - z;
}

/*
static float4 ContBoxfoldD(float4 z)
{
    float3 znew = z.xyz;
    float3 zsq = (float3)(znew.x * znew.x, znew.y * znew.y, znew.z * znew.z);
    zsq += (float3)(1, 1, 1);
    float3 res = (float3)(
        znew.x / sqrt(zsq.x), znew.y / sqrt(zsq.y), znew.z / sqrt(zsq.z));
    res *= sqrt(8.0f);
    res = znew - res;
    return (float4)(res.x, res.y, res.z, z.w);
}
*/

static float3 SpherefoldD(Cfg cfg, float3 z, float* dz)
{
    const float factor = fixed_radius_2 / clamp(dot(z, z), min_radius_2, fixed_radius_2);
    *dz *= factor;
    return z * factor;
}

/*
static float4 ContSpherefoldD(Cfg cfg, float4 z)
{
    z *= min_radius_2 / dot(z.xyz, z.xyz) + fixed_radius_2;
    return z;
}
*/

static float3 TScaleD(Cfg cfg, float3 z, float* dz)
{
    *dz *= fabs(scale);
    return z * scale;
}

static float3 TOffsetD(float3 z, float* dz, float3 offset)
{
    *dz += 1.0f;
    return z + offset;
}

static float3 Rotate(Cfg cfg, float3 z)
{
    float3 axis = normalize(LightPos1(cfg));
    float3 angle = rotation;
    return cos(angle) * z + sin(angle) * cross(z, axis) + (1 - cos(angle)) * dot(axis, angle) * axis;
}

static float3 MandelboxD(Cfg cfg, float3 z, float* dz, float3 offset, int* color)
{
    // z = ContBoxfoldD(cfg, z);
    // z = ContSpherefoldD(cfg, z);
    z = BoxfoldD(cfg, z);
    if (dot(z, z) < min_radius_2)
    {
        (*color)--;
    }
    else if (dot(z, z) < fixed_radius_2)
    {
        (*color)++;
    }
    z = Rotate(cfg, z);
    z = SpherefoldD(cfg, z, dz);
    z = TScaleD(cfg, z, dz);
    z = TOffsetD(z, dz, offset);
    return z;
}

static float DeSphere(float3 pos, float radius, float3 test)
{
    return length(test - pos) - radius;
}

static float DeMandelbox(Cfg cfg, float3 offset, float* color_data)
{
    float3 z = (float3)(offset.x, offset.y, offset.z);
    float dz = 1.0f;
    int n = max(max_iters, 1);
    int color = 0;
    do
    {
        z = MandelboxD(cfg, z, &dz, offset, &color);
    } while (dot(z, z) < bailout && --n);
    *color_data = color / 8.0f;
    return length(z) / dz;
}

static float Plane(float3 pos, float3 plane)
{
    return dot(pos, normalize(plane)) - length(plane);
}

static float De(Cfg cfg, float3 offset)
{
    float color_data;
    const float mbox = DeMandelbox(cfg, offset, &color_data);
    const float light1 = DeSphere(LightPos1(cfg), light_radius_1, offset);
    const float cut = Plane(offset, LightPos1(cfg));
    return max(min(light1, mbox), cut);
}

struct Material
{
    float3 color;
    float3 emissive;
    float specular;
};

// (r, g, b, spec)
static struct Material Material(Cfg cfg, float3 offset)
{
    float base_color;
    const float de = DeMandelbox(cfg, offset, &base_color);

    base_color *= surface_color_variance;
    base_color += surface_color_shift;
    float3 color = (float3)(
        sinpi(base_color), sinpi(base_color + 2.0f / 3.0f), sinpi(base_color + 4.0f / 3.0f));
    float specular = 0.0f;
    color = color * 0.5f + (float3)(0.5f, 0.5f, 0.5f);
    color = surface_color_saturation * (color - (float3)(1, 1, 1)) + (float3)(1, 1, 1);

    const float light1 = DeSphere(LightPos1(cfg), light_radius_1, offset);

    struct Material result;
    if (de < light1)
    {
        result.color = color * reflect_brightness;
        result.specular = specular;
        result.emissive = (float3)(0, 0, 0);
    }
    else
    {
        result.color = (float3)(1, 1, 1);
        result.specular = 0.0f;
        result.emissive = LightBrightness1(cfg);
    }

    return result;
}

static float Cast(Cfg cfg, struct Ray ray, const float quality, const float maxDist, float3* normal)
{
    float distance;
    float totalDistance = 0.0f;
    uint i = (uint)max(max_ray_steps, 1);
    do
    {
        distance = De(cfg, Ray_At(ray, totalDistance)) * de_multiplier;
        totalDistance += distance;
        i--;
    } while (totalDistance < maxDist && distance * quality > totalDistance && i > 0);

    // correction step
    if (distance * quality <= totalDistance)
    {
        totalDistance -= totalDistance / quality;
        for (int correctStep = 0; correctStep < 4; correctStep++)
        {
            distance = De(cfg, Ray_At(ray, totalDistance)) * de_multiplier;
            totalDistance += distance - totalDistance / quality;
        }
    }

    const float3 final = Ray_At(ray, totalDistance);
    const float delta = max(1e-6f, distance * 0.5f);  // aprox. 8.3x float epsilon
    const float dnpp = De(cfg, final + (float3)(-delta, delta, delta));
    const float dpnp = De(cfg, final + (float3)(delta, -delta, delta));
    const float dppn = De(cfg, final + (float3)(delta, delta, -delta));
    const float dnnn = De(cfg, final + (float3)(-delta, -delta, -delta));
    *normal = (float3)((dppn + dpnp) - (dnpp + dnnn),
                       (dppn + dnpp) - (dpnp + dnnn),
                       (dpnp + dnpp) - (dppn + dnnn));
    normal->x += (dot(*normal, *normal) == 0.0f);  // ensure nonzero
    *normal = normalize(*normal);
    return totalDistance;
}

static float3 Trace(
    Cfg cfg, struct Ray ray, const uint width, const uint height, struct Random* rand)
{
    float3 rayColor = (float3)(0, 0, 0);
    float3 reflectionColor = (float3)(1, 1, 1);
    float quality = quality_first_ray * ((width + height) / (2 * fov));

    for (int photonIndex = 0; photonIndex < num_ray_bounces; photonIndex++)
    {
        float3 normal;
        const float fog_dist = -native_log(Random_Next(rand)) * fog_distance;
        const float max_dist = min(max_ray_dist, fog_dist);
        const float distance = min(Cast(cfg, ray, quality, max_dist, &normal), fog_dist);

        if (distance >= max_ray_dist ||
            (photonIndex + 1 == num_ray_bounces && distance >= fog_dist))
        {
            // went out-of-bounds, or last fog ray didn't hit anything
            const float3 color = AmbientBrightness(cfg);
            rayColor += color * reflectionColor;
            break;
        }

        const float3 newPos = Ray_At(ray, min(distance, fog_dist));
        float3 newDir;

        if (distance >= fog_dist)
        {
            // hit fog, do fog calculations
            newDir = Random_Sphere(rand);
            reflectionColor *= fog_brightness;
        }
        else
        {
            // hit surface, do material calculations
            const struct Material material = Material(cfg, newPos);
            rayColor += reflectionColor * material.emissive;  // ~bling~!

            if (Random_Next(rand) < material.specular)
            {
                // specular
                newDir = ray.dir - 2 * dot(ray.dir, normal) * normal;
            }
            else
            {
                // diffuse
                newDir = Random_Lambertian(rand, normal);
                quality = quality_rest_ray;
            }
            const float incident_angle_weakening = dot(normal, newDir);
            reflectionColor *= incident_angle_weakening * material.color;
        }

        ray = new_Ray(newPos, newDir);

        if (dot(reflectionColor, reflectionColor) == 0)
        {
            break;
        }
    }
    return rayColor;
}

static float3 PreviewTrace(Cfg cfg, struct Ray ray, const uint width, const uint height)
{
    const float quality = quality_first_ray * ((width + height) / (2 * fov));
    const float max_dist = min(max_ray_dist, focal_distance * 10);
    float3 normal;
    // const float distance = Cast(cfg, ray, quality, max_dist, &normal);
    // const float value = distance / max_dist;
    // return (float3)(value);
    Cast(cfg, ray, quality, max_dist, &normal);
    return fabs(normal);
}

/*
static float3 GammaTest(int x, int y, int width, int height)
{
    const float centerValue = (float)x / width;
    const float offset = ((float)(height - y) / height) * (0.5f - fabs(centerValue - 0.5f));
    float result;
    if (x % 16 < 8)
    {
        result = centerValue;
    }
    else if (y % 2 == 0)
    {
        result = centerValue + offset;
    }
    else
    {
        result = centerValue - offset;
    }
    return (float3)(result, result, result);
}
// */

// Perfect, original sRGB
/*
static float GammaCompression(float value)
{
    if (value < 0.0031308f)
    {
        return 12.92f * value;
    }
    else
    {
        float a = 0.055f;
        return (1 + a) * powr(value, 1.0f / 2.4f) - a;
    }
}
// */
// http://mimosa-pudica.net/fast-gamma/
//*
static float GammaCompression(float value)
{
    const float a = 0.00279491f;
    const float b = 1.15907984f;
    // float c = b * native_rsqrt(1.0f + a) - 1.0f;
    const float c = 0.15746346551f;
    return (b * native_rsqrt(value + a) - c) * value;
}
// */
// gamma algorithm "an attempt was made"
/*
static float GammaCompression(float value)
{
    return sqrt(value);
}
// */

static uint PackPixel(Cfg cfg, float3 pixel)
{
    if (white_clamp)
    {
        const float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1)
        {
            pixel *= 1.0f / maxVal;
        }
        pixel = max(pixel, (float3)(0.0f, 0.0f, 0.0f));
    }
    else
    {
        pixel = clamp(pixel, 0.0f, 1.0f);
    }

    pixel.x = GammaCompression(pixel.x);
    pixel.y = GammaCompression(pixel.y);
    pixel.z = GammaCompression(pixel.z);

    pixel = pixel * 255;
    return ((uint)255 << 24) | ((uint)(uchar)pixel.x << 0) | ((uint)(uchar)pixel.y << 8) |
           ((uint)(uchar)pixel.z << 16);
}

// yay SOA
static float3 GetScratch(__global uchar* rawData, uint idx, uint size)
{
    const __global float* data = (__global float*)rawData;
    return (float3)(data[idx + size], data[idx + size * 2], data[idx + size * 3]);
}

static void SetScratch(__global uchar* rawData, uint idx, uint size, float3 value)
{
    __global float* data = (__global float*)rawData;
    data[idx + size] = value.x;
    data[idx + size * 2] = value.y;
    data[idx + size * 3] = value.z;
}

static struct Random GetRand(__global uchar* rawData, uint idx, uint size)
{
    const __global struct Random* data = (__global struct Random*)rawData;
    // this is scary: sizeof(random) is bigger than sizeof(float), so divide the
    // size multiplier by 2
    return data[idx + size * 2];
}

static void SetRand(__global uchar* rawData, uint idx, uint size, struct Random value)
{
    __global struct Random* data = (__global struct Random*)rawData;
    data[idx + size * 2] = value;
}

static void SetScreen(__global uchar* rawData, uint idx, uint size, uint value)
{
    __global uint* data = (__global uint*)rawData;
    data[idx] = value;
}

// type: -1 is preview, 0 is init, 1 is continue
__kernel void Main(__global uchar* data,
                   __global struct MandelboxCfg* cfg_global,
                   uint width,
                   uint height,
                   uint frame)
{
    const struct MandelboxCfg local_copy = *cfg_global;
    const Cfg cfg = &local_copy;

    const uint idx = (uint)get_global_id(0);
    const uint size = width * height;
    const uint x = idx % width;
    // flip image - in screen space, 0,0 is top-left, in 3d space, 0,0 is
    // bottom-left
    const uint y = height - (idx / width + 1);

    // out-of-bounds checks aren't needed, due to the ocl driver being awesome
    // and supporting exact-size launches

    const float3 oldColor = frame > 0 ? GetScratch(data, idx, size) : (float3)(0, 0, 0);

    struct Random rand = frame > 0 ? GetRand(data, idx, size) : new_Random(idx, frame, size);
    const struct Ray ray = Camera(cfg, x, y, width, height, &rand);
    const float3 colorComponents =
        preview ? PreviewTrace(cfg, ray, width, height) : Trace(cfg, ray, width, height, &rand);
    const float3 newColor = (colorComponents + oldColor * frame) / (frame + 1);
    // newColor = GammaTest(x, y, width, height);
    const uint packedColor = PackPixel(cfg, newColor);

    SetScratch(data, idx, size, newColor);
    SetRand(data, idx, size, rand);
    SetScreen(data, idx, size, packedColor);
}
