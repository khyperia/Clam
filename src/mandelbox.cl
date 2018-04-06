struct MandelboxCfg {
    float posX;
    float posY;
    float posZ;
    float lookX;
    float lookY;
    float lookZ;
    float upX;
    float upY;
    float upZ;
    float fov;
    float focalDistance;
    float Scale;
    float FoldingLimit;
    float FixedRadius2;
    float MinRadius2;
    float DofAmount;
    float LightPos1X;
    float LightPos1Y;
    float LightPos1Z;
    float LightBrightness1R;
    float LightBrightness1G;
    float LightBrightness1B;
    float LightPos2X;
    float LightPos2Y;
    float LightPos2Z;
    float LightBrightness2R;
    float LightBrightness2G;
    float LightBrightness2B;
    float AmbientBrightnessR;
    float AmbientBrightnessG;
    float AmbientBrightnessB;
    float ReflectBrightness;
    float Bailout;
    float DeMultiplier;
    float MaxRayDist;
    float QualityFirstRay;
    float QualityRestRay;
    int WhiteClamp;
    int MaxIters;
    int MaxRaySteps;
};

// When NumRayBounces is a dynamic variable in MandelboxCfg, the intel opencl
// runtime cannot vectorize the kernel.
#define NumRayBounces 3

typedef __private struct MandelboxCfg* Cfg;

static float3 LightPos1(Cfg cfg)
{
    return (float3)(cfg->LightPos1X, cfg->LightPos1Y, cfg->LightPos1Z);
}

static float3 LightPos2(Cfg cfg)
{
    return (float3)(cfg->LightPos2X, cfg->LightPos2Y, cfg->LightPos2Z);
}

static float3 LightBrightness1(Cfg cfg)
{
    return (float3)(cfg->LightBrightness1R, cfg->LightBrightness1G, cfg->LightBrightness1B) / cfg->ReflectBrightness;
}

static float3 LightBrightness2(Cfg cfg)
{
    return (float3)(cfg->LightBrightness2R, cfg->LightBrightness2G, cfg->LightBrightness2B) / cfg->ReflectBrightness;
}

static float3 AmbientBrightness(Cfg cfg)
{
    return (float3)(cfg->AmbientBrightnessR, cfg->AmbientBrightnessG, cfg->AmbientBrightnessB) / cfg->ReflectBrightness;
}

static float4 comp_mul4(float4 left, float4 right)
{
    return (float4)(
        left.x * right.x,
        left.y * right.y,
        left.z * right.z,
        left.w * right.w);
}

static float3 comp_mul3(float3 left, float3 right)
{
    return (float3)(
        left.x * right.x,
        left.y * right.y,
        left.z * right.z);
}

struct Random {
    ulong seed;
};

static float Random_Next(struct Random* this)
{
    this->seed = (this->seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return (this->seed >> 16) / 4294967296.0f;
}

static struct Random new_Random(uint idx, uint frame, uint global_size)
{
    struct Random result;
    ulong seed = idx + global_size * frame;
    result.seed = seed;
    for (int i = 0; i < 8; i++) {
        Random_Next(&result);
    }
    return result;
}

static float2 Random_Circle(struct Random* this)
{
    float2 polar = (float2)(Random_Next(this) * 2.0f, sqrt(Random_Next(this)));
    return (float2)(cospi(polar.x) * polar.y, sinpi(polar.x) * polar.y);
}

static float2 Random_Normal(struct Random* this)
{
    // Box-Muller transform
    // returns two normally-distributed independent variables
    float mul = sqrt(-2 * log2(Random_Next(this)));
    float angle = 2.0f * Random_Next(this);
    return mul * (float2)(cospi(angle), sinpi(angle));
}

static float3 Random_Sphere(struct Random* this)
{
    float2 temp = Random_Normal(this);
    float3 result = (float3)(temp.x, temp.y, Random_Normal(this).x);
    result.x += (dot(result, result) == 0.0f); // ensure nonzero
    return normalize(result);
}

static float3 Random_Lambertian(struct Random* this, float3 normal)
{
    float3 result = Random_Sphere(this);
    result = normalize(result + normal);
    return result;
}

struct Ray {
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
static float3 RayDir(float3 forward, float3 up, float2 screenCoords, float fov)
{
    screenCoords *= -fov;
    float len2 = dot(screenCoords, screenCoords);
    float3 look = (float3)(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);
    float3 right = cross(forward, up);
    return look.x * right + look.y * up + look.z * forward;
}

static float3 Ray_At(struct Ray this, float time)
{
    return this.pos + this.dir * time;
}

static void Ray_Dof(Cfg cfg, struct Ray* this, float focalPlane, struct Random* rand)
{
    float3 focalPosition = Ray_At(*this, focalPlane);
    float3 xShift = cross((float3)(0, 0, 1), this->dir);
    float3 yShift = cross(this->dir, xShift);
    float2 offset = Random_Circle(rand);
    float dofPickup = cfg->DofAmount;
    this->dir = normalize(this->dir + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    this->pos = focalPosition - this->dir * focalPlane;
}

static struct Ray Camera(Cfg cfg, uint x, uint y, uint width, uint height, struct Random* rand)
{
    float3 origin = (float3)(cfg->posX, cfg->posY, cfg->posZ);
    float3 look = (float3)(cfg->lookX, cfg->lookY, cfg->lookZ);
    float3 up = (float3)(cfg->upX, cfg->upY, cfg->upZ);
    float2 screenCoords = (float2)((float)x - (float)(width / 2), (float)y - (float)(height / 2));
    //screenCoords += (float2)(Random_Next(rand) - 0.5f, Random_Next(rand) - 0.5f);
    float fov = cfg->fov * 2 / (width + height);
    float3 direction = RayDir(look, up, screenCoords, fov);
    struct Ray result = new_Ray(origin, direction);
    Ray_Dof(cfg, &result, cfg->focalDistance, rand);
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

static float4 BoxfoldD(Cfg cfg, float4 z)
{
    float3 znew = z.xyz;
    znew = clamp(znew, -cfg->FoldingLimit, cfg->FoldingLimit) * 2.0f - znew;
    return (float4)(znew.x, znew.y, znew.z, z.w);
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

static float4 SpherefoldD(Cfg cfg, float4 z)
{
    z *= cfg->FixedRadius2 / clamp(dot(z.xyz, z.xyz), cfg->MinRadius2, cfg->FixedRadius2);
    return z;
}

/*
static float4 ContSpherefoldD(Cfg cfg, float4 z)
{
    z *= cfg->MinRadius2 / dot(z.xyz, z.xyz) + cfg->FixedRadius2;
    return z;
}
*/

static float4 TScaleD(Cfg cfg, float4 z)
{
    const float scale = cfg->Scale;
    const float4 mul = (float4)(scale, scale, scale, fabs(scale));
    return comp_mul4(z, mul);
}

static float4 TOffsetD(float4 z, float3 offset)
{
    return z + (float4)(offset.x, offset.y, offset.z, 1.0f);
}

static float4 MandelboxD(Cfg cfg, float4 z, float3 offset)
{
    // z = ContBoxfoldD(cfg, z);
    // z = ContSpherefoldD(cfg, z);
    z = BoxfoldD(cfg, z);
    z = SpherefoldD(cfg, z);
    z = TScaleD(cfg, z);
    z = TOffsetD(z, offset);
    return z;
}

static float DeSphere(float3 pos, float radius, float3 test)
{
    return length(test - pos) - radius;
}

static float DeMandelbox(Cfg cfg, float3 offset)
{
    float4 z = (float4)(offset.x, offset.y, offset.z, 1.0f);
    int n = max(cfg->MaxIters, 1);
    do {
        z = MandelboxD(cfg, z, offset);
    } while (dot(z.xyz, z.xyz) < cfg->Bailout && --n);
    return length(z.xyz) / z.w;
}

static float De(Cfg cfg, float3 offset)
{
    float mbox = DeMandelbox(cfg, offset);
    float light1 = DeSphere(LightPos1(cfg), 1.2f, offset);
    float light2 = DeSphere(LightPos2(cfg), 1.2f, offset);
    return min(min(light1, light2), mbox);
}

struct Material {
    float3 color;
    float specular;
    float3 emissive;
};

// (r, g, b, spec)
static struct Material Material(Cfg cfg, float3 offset)
{
    float4 z = (float4)(offset.x, offset.y, offset.z, 1.0f);
    int n = max(cfg->MaxIters, 1);
    float trap = 10000000000;
    do {
        z = MandelboxD(cfg, z, offset);
        trap = min(trap, length(trap));
    } while (dot(z.xyz, z.xyz) < cfg->Bailout && --n);
    float de = length(z.xyz) / z.w;

    trap = sin(log2(trap)) * 0.5f + 0.5f;
    float3 color = (float3)(trap, trap, trap);

    float light1 = DeSphere(LightPos1(cfg), 1.2f, offset);
    float light2 = DeSphere(LightPos2(cfg), 1.2f, offset);
    float minimum = min(min(light1, light2), de);

    struct Material result;
    if (minimum == de) {
        result.color = color * cfg->ReflectBrightness;
        result.specular = 0.2f;
        result.emissive = (float3)(0, 0, 0);
    } else if (minimum == light1) {
        result.color = (float3)(1, 1, 1);
        result.specular = 0.0f;
        result.emissive = LightBrightness1(cfg);
    } else if (minimum == light2) {
        result.color = (float3)(1, 1, 1);
        result.specular = 0.0f;
        result.emissive = LightBrightness2(cfg);
    }

    return result;
}

static float Cast(Cfg cfg, struct Ray ray, const float quality, const float maxDist, float3* normal)
{
    float distance;
    float totalDistance = 0.0f;
    int i = 0;
    const int maxSteps = max(cfg->MaxRaySteps, 1);
    const float deMultiplier = cfg->DeMultiplier;
    do {
        distance = De(cfg, Ray_At(ray, totalDistance)) * deMultiplier;
        totalDistance += distance;
        i++;
    } while (totalDistance < maxDist && distance * quality > totalDistance && i < maxSteps);
    float3 final = Ray_At(ray, totalDistance);
    float delta = max(1e-6f, distance * 0.5f); // aprox. 8.3x float epsilon
    float dnpp = De(cfg, final + (float3)(-delta, delta, delta));
    float dpnp = De(cfg, final + (float3)(delta, -delta, delta));
    float dppn = De(cfg, final + (float3)(delta, delta, -delta));
    float dnnn = De(cfg, final + (float3)(-delta, -delta, -delta));
    *normal = (float3)((dppn + dpnp) - (dnpp + dnnn),
        (dppn + dnpp) - (dpnp + dnnn),
        (dpnp + dnpp) - (dppn + dnnn));
    normal->x += (dot(*normal, *normal) == 0.0f); // ensure nonzero
    *normal = normalize(*normal);
    return totalDistance;
}

static float3 Trace(Cfg cfg, struct Ray ray, uint width, uint height, struct Random* rand)
{
    float3 rayColor = (float3)(0, 0, 0);
    float3 reflectionColor = (float3)(1, 1, 1);
    for (int i = 0; i < 3; i++) {
        const bool firstRay = i == 0;
        const float firstRayQuality = cfg->QualityFirstRay * ((width + height) / (2 * cfg->fov));

        const float quality = firstRay ? firstRayQuality : cfg->QualityRestRay;

        float3 normal;
        float distance = Cast(cfg, ray, quality, cfg->MaxRayDist, &normal);

        if (distance > cfg->MaxRayDist) {
            float first_reduce = firstRay ? 0.3f : 1.0f; // TODO
            float3 color = AmbientBrightness(cfg) * first_reduce;
            rayColor += comp_mul3(color, reflectionColor);
            break;
        }

        const float3 newPos = Ray_At(ray, distance);

        struct Material material = Material(cfg, newPos);
        rayColor += material.emissive;

        float3 newDir;
        if (Random_Next(rand) < material.specular) {
            // specular
            newDir = ray.dir - 2 * dot(ray.dir, normal) * normal;
        } else {
            // diffuse
            newDir = Random_Lambertian(rand, normal);
        }
        float incident_angle_weakening = dot(normal, newDir);
        reflectionColor *= incident_angle_weakening * material.color;
        ray = new_Ray(newPos, newDir);
    }
    return rayColor;
}

static float3 GammaTest(int x, int y, int width, int height)
{
    float centerValue = (float)x / width;
    float offset = ((float)(height - y) / height) * (0.5f - fabs(centerValue - 0.5f));
    float result;
    if (x % 16 < 8) {
        result = centerValue;
    } else if (y % 2 == 0) {
        result = centerValue + offset;
    } else {
        result = centerValue - offset;
    }
    return (float3)(result, result, result);
}

// Perfect, original sRGB
// static float GammaCompression(float value)
// {
//     if (value < 0.0031308f) {
//         return 12.92f * value;
//     } else {
//         float a = 0.055f;
//         return (1 + a) * powr(value, 1.0f / 2.4f) - a;
//     }
// }
// http://mimosa-pudica.net/fast-gamma/
static float GammaCompression(float value)
{
    const float a = 0.00279491f;
    const float b = 1.15907984f;
    //float c = b * native_rsqrt(1.0f + a) - 1.0f;
    const float c = 0.15746346551f;
    return (b * native_rsqrt(value + a) - c) * value;
}
// gamma algorithm "an attempt was made"
// static float GammaCompression(float value)
// {
//     return sqrt(value);
// }

static uint PackPixel(Cfg cfg, float3 pixel)
{
    if (cfg->WhiteClamp) {
        float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1) {
            pixel *= 1.0f / maxVal;
        }
        pixel = max(pixel, 0.0f);
    } else {
        pixel = clamp(pixel, 0.0f, 1.0f);
    }

    pixel.x = GammaCompression(pixel.x);
    pixel.y = GammaCompression(pixel.y);
    pixel.z = GammaCompression(pixel.z);

    pixel = pixel * 255;
    return ((uint)255 << 24) | ((uint)(uchar)pixel.x << 0) | ((uint)(uchar)pixel.y << 8) | ((uint)(uchar)pixel.z << 16);
}

// type: -1 is preview, 0 is init, 1 is continue
__kernel void Main(
    __global uchar* data,
    __global struct MandelboxCfg* cfg_global,
    uint width,
    uint height,
    uint frame)
{
    uint mem_offset = 0;
    __global uint* screenPixels = (__global uint*)(data + mem_offset);
    mem_offset += width * height * (uint)sizeof(uint);
    __global float3* scratchBuf_arg = (__global float3*)(data + mem_offset);
    struct MandelboxCfg local_copy = *cfg_global;
    struct MandelboxCfg* cfg = &local_copy;

    uint idx = (uint)get_global_id(0);
    uint x = idx % width;
    uint y = idx / width;
    if (y >= height) {
        return;
    }
    // flip image - in screen space, 0,0 is top-left, in 3d space, 0,0 is bottom-left
    y = height - (y + 1);
    struct Random rand = new_Random(idx, frame, width * height);
    struct Ray ray = Camera(cfg, x, y, width, height, &rand);
    float3 colorComponents = Trace(cfg, ray, width, height, &rand);
    __global float3* scratch = &scratchBuf_arg[idx];
    float3 oldColor = frame > 0 ? *scratch : (float3)(0, 0, 0);
    float3 newColor = (colorComponents + oldColor * frame) / (frame + 1);
    *scratch = newColor;

    //newColor = GammaTest(x, y, width, height);
    uint packedColor = PackPixel(cfg, newColor);
    screenPixels[idx] = packedColor;
}
