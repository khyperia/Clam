struct MandelboxCfg
{
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
    float LightPosX;
    float LightPosY;
    float LightPosZ;
    int WhiteClamp;
    float LightBrightnessHue;
    float LightBrightnessSat;
    float LightBrightnessVal;
    float AmbientBrightnessHue;
    float AmbientBrightnessSat;
    float AmbientBrightnessVal;
    float ReflectBrightnessHue;
    float ReflectBrightnessSat;
    float ReflectBrightnessVal;
    int MaxIters;
    float Bailout;
    float DeMultiplier;
    int RandSeedInitSteps;
    float MaxRayDist;
    int MaxRaySteps;
    int NumRayBounces;
    float QualityFirstRay;
    float QualityRestRay;
};

#ifndef __OPENCL_VERSION__
#define __global
#define __kernel
#endif

typedef __global struct MandelboxCfg *Cfg;

static float4 comp_mul(float4 left, float4 right)
{
    return (float4)(
        left.x * right.x,
        left.y * right.y,
        left.z * right.z,
        left.w * right.w);
}

struct Random
{
    ulong seed;
};

static unsigned int Random_MWC64X(struct Random *this)
{
    unsigned int c = this->seed >> 32, x = this->seed & 0xFFFFFFFF;
    this->seed = x * ((ulong)4294883355U) + c;
    return x ^ c;
}

static float Random_Next(struct Random *this)
{
    return (float)Random_MWC64X(this) / 4294967296.0f;
}

static struct Random new_Random(Cfg cfg, __global uint2 *randBufferArr, int idx, bool init)
{
    struct Random result;
    if (init)
    {
        result.seed = idx;
        for (int i = 0; ; i++)
        {
            const float bail = (float)i / cfg->RandSeedInitSteps - 1;
            const float test = Random_Next(&result);
            if (bail < test)
            {
                break;
            }
        }
    }
    else
    {
        uint2 randBuffer = randBufferArr[idx];
        result.seed = (ulong)randBuffer.x << 32 | (ulong)randBuffer.y;
    }
    return result;
}

static void Random_Save(struct Random this, __global uint2 *randBufferArr, int idx)
{
    randBufferArr[idx] = (uint2)((unsigned int)(this.seed >> 32), (unsigned int)this.seed);
}

static float2 Random_Circle(struct Random *this)
{
    float2 polar = (float2)(Random_Next(this) * 6.28318531f, sqrt(Random_Next(this)));
    return (float2)(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

static float2 Random_Normal(struct Random *this)
{
    // Box-Muller transform
    // returns two normally-distributed independent variables
    float mul = sqrt(-2 * log2(Random_Next(this)));
    float angle = 6.28318530718f * Random_Next(this);
    return mul * (float2)(cos(angle), sin(angle));
}

static float3 Random_Sphere(struct Random *this)
{
    float2 temp = Random_Normal(this);
    float3 result = (float3)(temp.x, temp.y, Random_Normal(this).x);
    result.x += (dot(result, result) == 0); // ensure nonzero
    return normalize(result);
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
static float3 RayDir(float3 forward, float3 up, float2 screenCoords, float fov)
{
    screenCoords *= -fov;
    float len2 = dot(screenCoords, screenCoords);
    float3 look =
        (float3)(2 * screenCoords.x, 2 * screenCoords.y, len2 - 1) / -(len2 + 1);
    float3 right = cross(forward, up);
    return look.x * right + look.y * up + look.z * forward;
}

static float3 Ray_At(struct Ray this, float time)
{
    return this.pos + this.dir * time;
}

static void Ray_Dof(Cfg cfg, struct Ray *this, float focalPlane, struct Random *rand)
{
    float3 focalPosition = Ray_At(*this, focalPlane);
    float3 xShift = cross((float3)(0, 0, 1), this->dir);
    float3 yShift = cross(this->dir, xShift);
    float2 offset = Random_Circle(rand);
    float dofPickup = cfg->DofAmount;
    this->dir = normalize(this->dir + offset.x * dofPickup * xShift + offset.y * dofPickup * yShift);
    this->pos = focalPosition - this->dir * focalPlane;
}

static struct Ray Camera(Cfg cfg, int x, int y, int screenX, int screenY, int width, int height, struct Random *rand)
{
    float3 origin = (float3)(cfg->posX, cfg->posY, cfg->posZ);
    float3 look = (float3)(cfg->lookX, cfg->lookY, cfg->lookZ);
    float3 up = (float3)(cfg->upX, cfg->upY, cfg->upZ);
    float2 screenCoords = (float2)((float)(x + screenX), (float)(y + screenY));
    screenCoords += (float2)(Random_Next(rand) - 0.5f, Random_Next(rand) - 0.5f);
    float fov = cfg->fov * 2 / (width + height);
    float3 direction = RayDir(look, up, screenCoords, fov);
    struct Ray result = new_Ray(origin, direction);
    Ray_Dof(cfg, &result, cfg->focalDistance, rand);
    return result;
}

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

static float4 BoxfoldD(Cfg cfg, float4 z)
{
    float3 znew = z.xyz;
    znew = clamp(znew, -cfg->FoldingLimit, cfg->FoldingLimit) * 2.0f - znew;
    return (float4)(znew.x, znew.y, znew.z, z.w);
}

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

static float4 SpherefoldD(Cfg cfg, float4 z)
{
    z *= cfg->FixedRadius2 / clamp(dot(z.xyz, z.xyz), cfg->MinRadius2, cfg->FixedRadius2);
    return z;
}

static float4 ContSpherefoldD(Cfg cfg, float4 z)
{
    z *= cfg->MinRadius2 / dot(z.xyz, z.xyz) + cfg->FixedRadius2;
    return z;
}

static float4 TScaleD(Cfg cfg, float4 z)
{
    const float scale = cfg->Scale;
    const float4 mul = (float4)(scale, scale, scale, fabs(scale));
    return comp_mul(z, mul);
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

static float De(Cfg cfg, float3 offset)
{
    float4 z = (float4)(offset.x, offset.y, offset.z, 1.0f);
    int n = max(cfg->MaxIters, 1);
    do
    {
        z = MandelboxD(cfg, z, offset);
    } while (dot(z.xyz, z.xyz) < cfg->Bailout && --n);
    return length(z.xyz) / z.w;
}

static float Cast(Cfg cfg, struct Ray ray, const float quality, const float maxDist, float3 *normal)
{
    float distance;
    float totalDistance = 0.0f;
    int i = 0;
    const int maxSteps = max(cfg->MaxRaySteps, 1);
    const float deMultiplier = cfg->DeMultiplier;
    do
    {
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
    normal->x += (dot(*normal, *normal) == 0); // ensure nonzero
    *normal = normalize(*normal);
    return totalDistance;
}

static float3 AmbientBrightness(Cfg cfg)
{
    return HueToRGB(cfg->AmbientBrightnessHue, cfg->AmbientBrightnessSat, cfg->AmbientBrightnessVal);
}

static float3 LightBrightness(Cfg cfg)
{
    return HueToRGB(cfg->LightBrightnessHue, cfg->LightBrightnessSat, cfg->LightBrightnessVal);
}

static float3 ReflectBrightness(Cfg cfg)
{
    return HueToRGB(cfg->ReflectBrightnessHue, cfg->ReflectBrightnessSat, cfg->ReflectBrightnessVal);
}

static float3 LightPos(Cfg cfg)
{
    return (float3)(cfg->LightPosX, cfg->LightPosY, cfg->LightPosZ);
}

static float
Specular(Cfg cfg, float3 incoming, float3 outgoing, float3 normal)
{
    const float3 half_vec = normalize(incoming + outgoing);
    const float dot_prod = fabs(dot(half_vec, normal));
    const float hardness = 128.0f;
    //const float base_reflection = 0.7f; // TODO: Make configurable.
    const float base_reflection = cfg->ReflectBrightnessVal;
    const float spec = pow(sinpi(dot_prod * 0.5f), hardness);
    return spec * (1 - base_reflection) + base_reflection;
}

static float3 Trace(Cfg cfg, struct Ray ray, int width, int height, struct Random *rand)
{
    float3 rayColor = (float3)(0, 0, 0);
    float reflectionColor = 1.0f;
    bool firstRay = true;
    const float maxDist = cfg->MaxRayDist;
    for (int i = 0; i < cfg->NumRayBounces; i++)
    {
        const float quality = firstRay ? cfg->QualityFirstRay *
            ((width + height) / (2 * cfg->fov)) : cfg->QualityRestRay;
        float3 normal;
        const float distance = Cast(cfg, ray, quality, maxDist, &normal);
        if (distance > maxDist)
        {
            float3 color = firstRay
                ? (float3)(0.3f, 0.3f, 0.3f)
                : AmbientBrightness(cfg);
            rayColor += color * reflectionColor;
            break;
        }
        // incorporates lambertian lighting
        const float3 newDir = normalize(Random_Sphere(rand) + normal);
        const float3 newPos = Ray_At(ray, distance);

        // direct lighting
        // TODO: Soft shadows, shuffle lightPos a bit
        const float3 toLightVec = LightPos(cfg) - newPos;
        const float lightDist = length(toLightVec);
        const float3 toLight = toLightVec * (1 / lightDist);
        const float normalDotProd = dot(normal, toLight);
        if (normalDotProd > 0)
        {
            float3 discard_normal;
            const float distance = Cast(cfg, new_Ray(newPos, toLight), quality, lightDist, &discard_normal);
            if (distance >= lightDist)
            {
                const float dimmingFactor =
                    (normalDotProd * reflectionColor * Specular(cfg, toLight, -ray.dir, normal)) /
                    (lightDist * lightDist);
                rayColor += LightBrightness(cfg) * dimmingFactor;
            }
        }
        reflectionColor *= Specular(cfg, newDir, -ray.dir, normal);
        ray = new_Ray(newPos, newDir);
        firstRay = false;
    }
    return rayColor;
}

static unsigned int PackPixel(Cfg cfg, float3 pixel)
{
    const float gamma_correct = 1 / 2.2f;
    pixel.x = pow(pixel.x, gamma_correct);
    pixel.y = pow(pixel.y, gamma_correct);
    pixel.z = pow(pixel.z, gamma_correct);
    if (cfg->WhiteClamp)
    {
        float maxVal = max(max(pixel.x, pixel.y), pixel.z);
        if (maxVal > 1)
        {
            pixel *= 1.0f / maxVal;
        }
    }
    else
    {
        pixel = clamp(pixel, 0.0f, 1.0f);
    }
    pixel = pixel * 255;
    return ((unsigned int)255 << 24) | ((unsigned int)(unsigned char)pixel.x << 16) |
        ((unsigned int)(unsigned char)pixel.y << 8) | ((unsigned int)(unsigned char)pixel.z);
}

// type: -1 is preview, 0 is init, 1 is continue
__kernel void Main(
    __global unsigned int *screenPixels,
    __global struct MandelboxCfg *cfg,
    __global float4 *scratchBuf_arg,
    __global uint2 *randBuf_arg,
    int screenX,
    int screenY,
    int width,
    int height,
    int frame
)
{
    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;
    if (y >= height)
    {
        return;
    }
    // flip image - in screen space, 0,0 is top-left, in 3d space, 0,0 is bottom-left
    y = height - (y + 1);
    struct Random rand = new_Random(cfg, randBuf_arg, idx, frame <= 0);
    struct Ray ray = Camera(cfg, x, y, screenX, screenY, width, height, &rand);
    float3 color = Trace(cfg, ray, width, height, &rand);
    __global float4 *scratch = &scratchBuf_arg[idx];
    float4 oldColor = frame > 0 ? *scratch : (float4)(0, 0, 0, 0);
    float newWeight = oldColor.w + 1;
    float3 newColor = (color + oldColor.xyz * oldColor.w) / newWeight;
    *scratch = (float4)(newColor.x, newColor.y, newColor.z, newWeight);

    int packedColor = PackPixel(cfg, newColor);
    screenPixels[idx] = packedColor;
    Random_Save(rand, randBuf_arg, idx);
}
