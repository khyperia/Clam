#version 430
layout(rgba32f, binding = 0) uniform image2D img_output;
layout(rgba32f, binding = 1) uniform image2D scratch;
layout(r32ui, binding = 2) uniform uimage2D randbuf;
// precision highp float;

#define PI  3.1415926538
#define TAU 6.28318530718
#define FLT_MAX 1E+37
#define UINT_MAX 0xFFFFFFFFU

uniform float pos_x;                     // 0.0 1.0
uniform float pos_y;                     // 0.0 1.0
uniform float pos_z;                     // 5.0 1.0
uniform float look_x;                    // 0.0 1.0
uniform float look_y;                    // 0.0 1.0
uniform float look_z;                    // -1.0 1.0
uniform float up_x;                      // 0.0 1.0
uniform float up_y;                      // 1.0 1.0
uniform float up_z;                      // 0.0 1.0
uniform float fov;                       // 1.0 -1.0
uniform float focal_distance;            // 3.0 -1.0
uniform float scale;                     // -2.0 0.5 const
uniform float folding_limit;             // 1.0 -0.5 const
uniform float fixed_radius_2;            // 1.0 -0.5 const
uniform float min_radius_2;              // 0.125 -0.5 const
uniform float dof_amount;                // 0.01 -1.0
uniform float fog_distance;              // 10.0 -1.0
uniform float fog_brightness;            // 1.0 0.5
uniform float light_pos_1_x;             // 3.0 1.0
uniform float light_pos_1_y;             // 3.5 1.0
uniform float light_pos_1_z;             // 2.5 1.0
uniform float light_radius_1;            // 1.0 -0.5
uniform float light_brightness_1_hue;    // 0.0 0.25
uniform float light_brightness_1_sat;    // 0.4 -1.0
uniform float light_brightness_1_val;    // 4.0 -1.0
uniform float ambient_brightness_hue;    // 0.65 0.25
uniform float ambient_brightness_sat;    // 0.2 -1.0
uniform float ambient_brightness_val;    // 0.5 -1.0
uniform float surface_color_variance;    // 0.0625 -0.25
uniform float surface_color_shift;       // 0.0 0.125
uniform float surface_color_saturation;  // 1.0 0.125
uniform float surface_color_value;       // 1.0 0.125
uniform float surface_color_specular;    // 0.0 0.125
uniform float plane_x;                   // 3.0 1.0
uniform float plane_y;                   // 3.5 1.0
uniform float plane_z;                   // 2.5 1.0
uniform float rotation;                  // 0.0 0.125
uniform float bailout;                   // 64.0 -1.0 const
uniform float bailout_normal;            // 1024.0 -1.0 const
uniform float de_multiplier;             // 0.9375 0.125 const
uniform float max_ray_dist;              // 16.0 -0.5 const
uniform float quality_first_ray;         // 2.0 -0.5 const
uniform float quality_rest_ray;          // 64.0 -0.5 const
uniform float gamma;                     // 0.0 0.25 const
uniform float fov_left;                  // -1.0 1.0 const
uniform float fov_right;                 // 1.0 1.0 const
uniform float fov_top;                   // 1.0 1.0 const
uniform float fov_bottom;                // -1.0 1.0 const
uniform uint max_iters;                  // 20 const
uniform uint max_ray_steps;              // 256 const
uniform uint num_ray_bounces;            // 4 const
uniform uint width;
uniform uint height;
uniform uint frame;

vec3 HueToRGB(float hue, float saturation, float value)
{
    hue = mod(hue, 1.0f);
    hue *= 3;
    float frac = mod(hue, 1.0f);
    vec3 color;
    switch (int(hue))
    {
        case 0:
            color = vec3(1 - frac, frac, 0);
            break;
        case 1:
            color = vec3(0, 1 - frac, frac);
            break;
        case 2:
            color = vec3(frac, 0, 1 - frac);
            break;
        default:
            color = vec3(1, 1, 1);
            break;
    }
    saturation = value * (1 - saturation);
    color = color * (value - saturation) + vec3(saturation, saturation, saturation);
    return color;
}

vec3 LightPos1()
{
    return vec3(light_pos_1_x, light_pos_1_y, light_pos_1_z);
}

vec3 PlanePos()
{
    return vec3(plane_x, plane_y, plane_z);
}

vec3 LightBrightness1()
{
    return HueToRGB(light_brightness_1_hue,
                    light_brightness_1_sat,
                    light_brightness_1_val) /
           surface_color_value;
}

vec3 AmbientBrightness()
{
    return HueToRGB(ambient_brightness_hue,
                    ambient_brightness_sat,
                    ambient_brightness_val) /
           surface_color_value;
}

struct Random
{
    uint seed;
};

uint xorshift32(inout uint x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

float Random_Next(inout Random this_)
{
    return float(xorshift32(this_.seed.x)) / UINT_MAX;
}

/*
struct Random new_Random(uint idx, uint frame, uint global_size)
{
    struct Random result;
    result.seed = (ulong)idx + (ulong)global_size * (ulong)frame;
    for (int i = 0; i < 8; i++)
    {
        Random_Next(result);
    }
    return result;
}
*/

void Random_Init(inout Random this_)
{
    this_.seed += gl_GlobalInvocationID.x + 10;
    for (int i = 0; i < 8; i++)
    {
        Random_Next(this_);
    }
}

vec2 Random_Disk(inout Random this_)
{
    const vec2 polar = vec2(Random_Next(this_), sqrt(Random_Next(this_)));
    return vec2(cos(TAU * polar.x) * polar.y, sin(TAU * polar.x) * polar.y);
}

vec3 Random_Sphere(inout Random this_)
{
    const float theta = Random_Next(this_);
    const float cosphi = 2 * Random_Next(this_) - 1;
    const float sinphi = sqrt(1 - cosphi * cosphi);
    const float x = sinphi * cos(TAU * theta);
    const float y = sinphi * sin(TAU * theta);
    const float z = cosphi;
    return vec3(x, y, z);
}

vec3 Random_Ball(inout Random this_)
{
    return pow(Random_Next(this_), 1.0/3.0) * Random_Sphere(this_);
}

vec3 Random_Lambertian(inout Random this_, vec3 normal)
{
    return normalize(Random_Ball(this_) + normal);
}

struct Ray
{
    vec3 pos;
    vec3 dir;
};

#ifdef VR
float Remap(float value, float iMin, float iMax, float oMin, float oMax)
{
    float slope = (oMax - oMin) / (iMax - iMin);
    float offset = oMin - iMin * slope;
    return value * slope + offset;
}

vec3 RayDir(vec3 forward, vec3 up, vec2 screenCoords)
{
    float x = Remap(screenCoords.x, 0, 1, fov_left, fov_right);
    float y = Remap(screenCoords.y, 0, 1, fov_top, fov_bottom);
    float z = 1.0f;
    vec3 dir = normalize(vec3(x, y, z));
    vec3 right = cross(forward, up);
    return dir.x * right + dir.y * up + dir.z * forward;
}
#else
// http://en.wikipedia.org/wiki/Stereographic_projection
vec3 RayDir(vec3 forward, vec3 up, vec2 screenCoords, float calcFov)
{
    screenCoords *= -calcFov;
    const float len2 = dot(screenCoords, screenCoords);
    const vec3 look = vec3(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);
    const vec3 right = cross(forward, up);
    return look.x * right + look.y * up + look.z * forward;
}
#endif

vec3 Ray_At(Ray this_, float time)
{
    return this_.pos + this_.dir * time;
}

void Ray_Dof(inout Ray this_, float focalPlane, inout Random rand)
{
    const vec3 focalPosition = Ray_At(this_, focalPlane);
    // Normalize because the vectors aren't perpendicular
    const vec3 xShift = normalize(cross(vec3(0, 0, 1), this_.dir));
    const vec3 yShift = cross(this_.dir, xShift);
    const vec2 offset = Random_Disk(rand);
    const float dofPickup = dof_amount;
    this_.dir =
        normalize(this_.dir + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    this_.pos = focalPosition - this_.dir * focalPlane;
}

Ray Camera(uint x, uint y, uint width, uint height, inout Random rand)
{
    const vec3 origin = vec3(pos_x, pos_y, pos_z);
    const vec3 look = vec3(look_x, look_y, look_z);
    const vec3 up = vec3(up_x, up_y, up_z);
#ifdef VR
    const vec2 screenCoords = vec2(float(x) / float(width), float(y) / float(height));
    const vec3 direction = RayDir(look, up, screenCoords);
#else
    const vec2 antialias = vec2(Random_Next(rand), Random_Next(rand)) - vec2(0.5f, 0.5f);
    const vec2 screenCoords =
        vec2(float(x) - float(width) / 2.0, float(y) - float(height) / 2.0) + antialias;
    const float calcFov = fov * 2.0 / float(width + height);
    const vec3 direction = RayDir(look, up, screenCoords, calcFov);
#endif
    Ray result = Ray(origin, direction);
#ifndef VR
    Ray_Dof(result, focal_distance, rand);
#endif
    return result;
}

vec3 Mandelbulb(vec3 z, inout float dz, const float Power)
{
    const float r = length(z.xyz);
    // convert to polar coordinates
    float theta = asin(z.z / r);
    float phi = atan(z.y, z.x);
    dz = pow(r, Power - 1.0f) * Power * dz + 1.0f;
    // scale and rotate the point
    float zr = pow(r, Power);
    theta = theta * Power;
    phi = phi * Power;
    // convert back to cartesian coordinates
    return zr * vec3(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
}

vec3 BoxfoldD(vec3 z)
{
    return clamp(z, -folding_limit, folding_limit) * 2.0f - z;
}

vec3 ContBoxfoldD(vec3 z)
{
    vec3 znew = z.xyz;
    vec3 zsq = vec3(znew.x * znew.x, znew.y * znew.y, znew.z * znew.z);
    zsq += vec3(1, 1, 1);
    vec3 res = vec3(znew.x / sqrt(zsq.x), znew.y / sqrt(zsq.y), znew.z / sqrt(zsq.z));
    res *= sqrt(8.0f);
    res = znew - res;
    return vec3(res.x, res.y, res.z);
}

vec3 SpherefoldD(vec3 z, inout float dz)
{
    const float factor =
        fixed_radius_2 / clamp(dot(z, z), min_radius_2, fixed_radius_2);
    dz *= factor;
    return z * factor;
}

vec3 ContSpherefoldD(vec3 z, inout float dz)
{
    const float mul = min_radius_2 / dot(z, z) + fixed_radius_2;
    z *= mul;
    dz *= mul;
    return z;
}

vec3 TScaleD(vec3 z, inout float dz)
{
    dz *= abs(scale);
    return z * scale;
}

vec3 TOffsetD(vec3 z, inout float dz, vec3 offset)
{
    dz += 1.0f;
    return z + offset;
}

vec3 Rotate(vec3 z)
{
    vec3 axis = normalize(PlanePos());
    float angle = rotation;
    return cos(angle) * z + sin(angle) * cross(z, axis) +
           (1 - cos(angle)) * dot(axis, vec3(angle)) * axis;
}

vec3 MandelboxD(vec3 z, inout float dz, vec3 offset, inout int color)
{
#ifdef CONT_FOLD
    z = ContBoxfoldD(z);
#else
    z = BoxfoldD(z);
#endif
    if (dot(z, z) < min_radius_2)
    {
        color--;
    }
    else if (dot(z, z) < fixed_radius_2)
    {
        color++;
    }
#ifdef ROTATE
    z = Rotate(z);
#endif
#ifdef CONT_FOLD
    z = ContSpherefoldD(z, dz);
#else
    z = SpherefoldD(z, dz);
#endif
    z = TScaleD(z, dz);
    z = TOffsetD(z, dz, offset);
    return z;
}

vec3 MandelbulbD(vec3 z, inout float dz, vec3 offset, inout int color)
{
    z = Mandelbulb(z, dz, abs(scale));
    if (color == 0)
    {
        color = 1 << 30;
    };
    color = min(color, int(dot(z, z) * 1000));
#ifdef ROTATE
    z = Rotate(z);
#endif
    return z + offset;
}

float DeSphere(vec3 pos, float radius, vec3 test)
{
    if (radius == 0)
    {
        return FLT_MAX;
    }
    return length(test - pos) - radius;
}

float DeMandelbox(vec3 offset, bool isNormal, inout int color)
{
    vec3 z = vec3(offset.x, offset.y, offset.z);
    float dz = 1.0f;
    uint n = max(max_iters, 1);
    float bail = isNormal ? bailout_normal : bailout;
    do
    {
        z = MandelboxD(z, dz, offset, color);
    } while (dot(z, z) < bail * bail && --n != 0);
    return length(z) / dz;
}

float DeMandelbulb(vec3 offset, inout int color)
{
    vec3 z = vec3(offset.x, offset.y, offset.z);
    float dz = 1.0f;
    uint n = max(max_iters, 1);
    do
    {
        z = MandelbulbD(z, dz, offset, color);
    } while (dot(z, z) < 4 && --n != 0);
    float r = length(z);
    return 0.5f * log(r) * r / dz;
}

float DeFractal(vec3 offset, bool isNormal, inout int color)
{
#ifdef MANDELBULB
    return DeMandelbulb(offset, color);
#else
    return DeMandelbox(offset, isNormal, color);
#endif
}

float Plane(vec3 pos, vec3 plane)
{
    return dot(pos, normalize(plane)) - length(plane);
}

float De(vec3 offset, bool isNormal)
{
    int color;
    const float mbox = DeFractal(offset, isNormal, color);
    const float light1 = DeSphere(LightPos1(), light_radius_1, offset);
#ifdef PLANE
    const float cut = Plane(offset, PlanePos());
    return min(light1, max(mbox, cut));
#else
    return min(light1, mbox);
#endif
}

struct Material
{
    vec3 color;
    vec3 normal;
    vec3 emissive;
    float specular;
};

// (r, g, b, spec)
Material GetMaterial(vec3 offset)
{
    int raw_color_data = 0;
    const float de = DeFractal(offset, true, raw_color_data);

    const float light1 = DeSphere(LightPos1(), light_radius_1, offset);

    Material result;
    if (de < light1)
    {
        const float hue = float(raw_color_data) * surface_color_variance + surface_color_shift;
        const vec3 color = HueToRGB(hue, surface_color_saturation, surface_color_value);
        result.color = color;
        result.specular = surface_color_specular;
        result.emissive = vec3(0, 0, 0);
    }
    else
    {
        result.color = vec3(1, 1, 1);
        result.specular = 0.0f;
        result.emissive = LightBrightness1();
    }

    const float delta = max(1e-6f, de * 0.5f);  // aprox. 8.3x float epsilon
#ifdef CUBE_NORMAL
    const float dppp = De(offset + vec3(+delta, +delta, +delta), true);
    const float dppn = De(offset + vec3(+delta, +delta, -delta), true);
    const float dpnp = De(offset + vec3(+delta, -delta, +delta), true);
    const float dpnn = De(offset + vec3(+delta, -delta, -delta), true);
    const float dnpp = De(offset + vec3(-delta, +delta, +delta), true);
    const float dnpn = De(offset + vec3(-delta, +delta, -delta), true);
    const float dnnp = De(offset + vec3(-delta, -delta, +delta), true);
    const float dnnn = De(offset + vec3(-delta, -delta, -delta), true);
    result.normal = vec3((dppp + dppn + dpnp + dpnn) - (dnpp + dnpn + dnnp + dnnn),
                             (dppp + dppn + dnpp + dnpn) - (dpnp + dpnn + dnnp + dnnn),
                             (dppp + dpnp + dnpp + dnnp) - (dppn + dpnn + dnpn + dnnn));
#else
    const float dnpp = De(offset + vec3(-delta, delta, delta), true);
    const float dpnp = De(offset + vec3(delta, -delta, delta), true);
    const float dppn = De(offset + vec3(delta, delta, -delta), true);
    const float dnnn = De(offset + vec3(-delta, -delta, -delta), true);
    result.normal = vec3((dppn + dpnp) - (dnpp + dnnn),
                             (dppn + dnpp) - (dpnp + dnnn),
                             (dpnp + dnpp) - (dppn + dnnn));
#endif
    result.normal.x += (dot(result.normal, result.normal) == 0.0f ? 1.0 : 0.0);  // ensure nonzero
    result.normal = normalize(result.normal);

    return result;
}

float Cast(Ray ray, const float quality, const float maxDist)
{
    float distance;
    float totalDistance = 0.0f;
    uint i = uint(max(max_ray_steps, 1));
    do
    {
        distance = De(Ray_At(ray, totalDistance), false) * de_multiplier;
        totalDistance += distance;
        i--;
    } while (totalDistance < maxDist && distance * quality > totalDistance && i > 0);

    // correction step
    if (distance * quality <= totalDistance)
    {
        totalDistance -= totalDistance / quality;
        for (int correctStep = 0; correctStep < 4; correctStep++)
        {
            distance = De(Ray_At(ray, totalDistance), false) * de_multiplier;
            totalDistance += distance - totalDistance / quality;
        }
    }
    return totalDistance;
}

vec3 Trace(
    Ray ray, const uint width, const uint height, inout Random rand)
{
    vec3 rayColor = vec3(0, 0, 0);
    vec3 reflectionColor = vec3(1, 1, 1);
    float quality = quality_first_ray * (float(width + height) / float(2 * fov));

    for (int photonIndex = 0; photonIndex < num_ray_bounces; photonIndex++)
    {
        const float fog_dist =
            fog_distance == 0 ? FLT_MAX : -log(Random_Next(rand)) * fog_distance;
        const float max_dist = min(max_ray_dist, fog_dist);
        const float distance = min(Cast(ray, quality, max_dist), fog_dist);

        if (distance >= max_ray_dist ||
            (photonIndex + 1 == num_ray_bounces && distance >= fog_dist))
        {
            // went out-of-bounds, or last fog ray didn't hit anything
            const vec3 color = AmbientBrightness();
            rayColor += color * reflectionColor;
            break;
        }

        const vec3 newPos = Ray_At(ray, min(distance, fog_dist));

        vec3 newDir;
        Material material;

        bool is_fog = (distance >= fog_dist);
        if (is_fog)
        {
            // hit fog, do fog calculations
            newDir = Random_Sphere(rand);
            reflectionColor *= fog_brightness;
            material.color = vec3(1, 1, 1);
            material.normal = vec3(0, 0, 0);
            material.specular = 0;
            material.emissive = vec3(0, 0, 0);
        }
        else
        {
            // hit surface, do material calculations
            material = GetMaterial(newPos);
            rayColor += reflectionColor * material.emissive;  // ~bling~!

            /*
                // Assuming for example:
                //   diffuse = albedo / PI;
                //   specular = my_favorite_glossy_brdf(in_dir, out_dir, roughness);
                //   fresnel = f0 + (1.0 - f0) * pow(1.0 - dot(E, H), 5.0);
                //   total_surface_reflection = fresnel
                color = lightIntensity * dot(L, N) * Lerp(diffuse, specular,
               total_surface_reflection);
            */
            float cosTheta = -dot(ray.dir, material.normal);
            float fresnel =
                material.specular + (1.0f - material.specular) * pow(1.0 - cosTheta, 5);
            if (Random_Next(rand) < fresnel)
            {
                // specular
                newDir = ray.dir + 2.0 * cosTheta * material.normal;
                // material.color = vec3(1.0, 1.0, 1.0);
            }
            else
            {
                // diffuse
                newDir = Random_Lambertian(rand, material.normal);
                quality = quality_rest_ray;
            }
        }

        if (light_radius_1 == 0)
        {
            vec3 lightPos = LightPos1();
            vec3 dir = lightPos - newPos;
            float light_dist = length(dir);
            dir = normalize(dir);
            if (is_fog || dot(dir, material.normal) > 0)
            {
                float dist = Cast(Ray(newPos, dir), quality_rest_ray, light_dist);
                if (dist >= light_dist)
                {
                    float prod = is_fog ? 1.0f : dot(material.normal, dir);
                    // Fixes tiny lil bright pixels
                    float fixed_dist = max(distance / 2.0f, light_dist);
                    vec3 color =
                        prod / (fixed_dist * fixed_dist) * material.color * LightBrightness1();
                    rayColor += reflectionColor * color;  // ~bling~!
                }
            }
        }

        const float incident_angle_weakening = (is_fog ? 1.0f : dot(material.normal, newDir));
        reflectionColor *= incident_angle_weakening * material.color;

        ray = Ray(newPos, newDir);

        if (dot(reflectionColor, reflectionColor) == 0)
        {
            break;
        }
    }
    return rayColor;
}

vec3 PreviewTrace(Ray ray, const uint width, const uint height)
{
    const float quality = quality_first_ray * (float(width + height) / float(2 * fov));
    const float max_dist = min(max_ray_dist, focal_distance * 10);
    const float distance = Cast(ray, quality, max_dist);
#ifdef PREVIEW_NORMAL
    const vec3 pos = Ray_At(ray, distance);
    return abs(GetMaterial(pos).normal);
#else
    const float value = distance / max_dist;
    return vec3(value);
#endif
}

vec3 GammaTest(uint x, uint y, uint width, uint height)
{
    const float centerValue = float(x) / float(width);
    const float offset = float((height - y)) / float(height) * (0.5f - abs(centerValue - 0.5f));
    float result;
    const int column_width = 8;
    if (x % (column_width * 2) < column_width)
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
    return vec3(result, result, result);
}

// Perfect, original sRGB
float GammaSRGB(float value)
{
    if (value < 0.0031308f)
    {
        return 12.92f * value;
    }
    else
    {
        float a = 0.055f;
        return (1 + a) * pow(value, 1.0f / 2.4f) - a;
    }
}

// http://mimosa-pudica.net/fast-gamma/
float GammaFast(float value)
{
    const float a = 0.00279491f;
    const float b = 1.15907984f;
    // float c = b * rsqrt(1.0f + a) - 1.0f;
    const float c = 0.15746346551f;
    return (b / sqrt(value + a) - c) * value;
}

float GammaPow(float value, float power)
{
    return pow(value, power);
}

float GammaCompression(float value)
{
    float gam = gamma;
    if (gam < 0)
    {
        return GammaSRGB(value);
    }
    else if (gam == 0)
    {
        return GammaFast(value);
    }
    else
    {
        return GammaPow(value, 1.0 / gam);
    }
}

vec3 PackPixel(vec3 pixel)
{
    if (isnan(pixel.x) || isnan(pixel.y) || isnan(pixel.z))
    {
        return vec3(1.0f, 0.0f, 1.0f);
    }

#ifdef CLAMP_PRESERVE_COLOR
    const float maxVal = max(max(pixel.x, pixel.y), pixel.z);
    if (maxVal > 1)
    {
        pixel *= 1.0f / maxVal;
    }
    pixel = max(pixel, vec3(0.0f, 0.0f, 0.0f));
#else
    pixel = clamp(pixel, 0.0f, 1.0f);
#endif

    pixel.x = GammaCompression(pixel.x);
    pixel.y = GammaCompression(pixel.y);
    pixel.z = GammaCompression(pixel.z);

    return pixel;
}

vec3 GetScratch(uint x, uint y)
{
    return imageLoad(scratch, ivec2(x, y)).xyz;
}

void SetScratch(uint x, uint y, vec3 value)
{
    imageStore(scratch, ivec2(x, y), vec4(value, 1.0));
}

Random GetRand(uint x, uint y)
{
    uint value = imageLoad(randbuf, ivec2(x, y)).x;
    Random rand = Random(value);
    Random_Init(rand);
    return rand;
}

void SetRand(uint x, uint y, Random value)
{
    imageStore(randbuf, ivec2(x, y), uvec4(value.seed, 0, 0, 0));
}

void SetScreen(uint x, uint y, vec3 value)
{
#ifdef VR
    value *= 255;
#endif
    imageStore(img_output, ivec2(x, y), vec4(value, 1.0));
}

void main()
{
    const uint idx = gl_GlobalInvocationID.x;
    const uint size = width * height;
    if (idx >= size) { return; }
    const uint x = idx % width;
    const uint y = idx / width;

    const vec3 oldColor = frame > 0 ? GetScratch(x, y) : vec3(0, 0, 0);

    Random rand = GetRand(x, y);
    const Ray ray = Camera(x, y, width, height, rand);
#ifdef PREVIEW
    const vec3 colorComponents = PreviewTrace(ray, width, height);
#else
    const vec3 colorComponents = Trace(ray, width, height, rand);
#endif
#ifdef GAMMA_TEST
    const vec3 newColor = GammaTest(x, y, width, height);
#else
    const vec3 newColor = (colorComponents + oldColor * frame) / (frame + 1);
    SetScratch(x, y, newColor);
    SetRand(x, y, rand);
#endif
    vec3 packedColor = PackPixel(newColor);

    SetScreen(x, y, packedColor);
}
