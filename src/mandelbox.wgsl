// layout(local_size_x = 64) in;
@group(0) @binding(0) 
var<storage,read_write> img: array<vec4<f32>>;
@group(0) @binding(1) 
var<storage,read_write> randbuf: array<u32>;
@group(0) @binding(3) 
var samp: sampler;
@group(0) @binding(4) 
var sky: texture_2d<f32>;

struct Data {
    pos: vec4<f32>,
    look: vec4<f32>,
    up: vec4<f32>,
    fov: f32,
    focal_distance: f32,
    scale: f32,
    folding_limit: f32,
    fixed_radius_2: f32,
    min_radius_2: f32,
    dof_amount: f32,
    bloom_amount: f32,
    bloom_size: f32,
    fog_distance: f32,
    fog_brightness: f32,
    exposure: f32,
    surface_color_variance: f32,
    surface_color_shift: f32,
    surface_color_saturation: f32,
    surface_color_value: f32,
    surface_color_gloss: f32,
    plane: vec4<f32>,
    rotation: f32,
    bailout: f32,
    bailout_normal: f32,
    de_multiplier: f32,
    max_ray_dist: f32,
    quality_first_ray: f32,
    quality_rest_ray: f32,
    gamma: f32,
    fov_left: f32,
    fov_right: f32,
    fov_top: f32,
    fov_bottom: f32,
    max_iters: u32,
    max_ray_steps: u32,
    num_ray_bounces: u32,
    gamma_test: u32,
    width: u32,
    height: u32,
    frame: u32,
}

@group(0) @binding(2) 
var<uniform> data: Data;

fn HueToRGB(huep: f32, saturationp: f32, value: f32) -> vec3<f32> {
    var hue = huep;
    hue = hue % 1.0;
    hue *= 3.0;
    var frac = hue % 1.0;
    var color: vec3<f32>;
    switch u32(hue) {
        case 0u: { color = vec3<f32>(1.0 - frac, frac, 0.0); }
        case 1u: { color = vec3<f32>(0.0, 1.0 - frac, frac); }
        case 2u: { color = vec3<f32>(frac, 0.0, 1.0 - frac); }
        default: { color = vec3<f32>(1.0, 1.0, 1.0); }
    }
    var saturation = saturationp;
    saturation = value * (1.0 - saturation);
    color = color * (value - saturation) + vec3(saturation, saturation, saturation);
    return color;
}

fn SampleSphericalMap(v: vec3<f32>) -> vec2<f32> {
    var invAtan = vec2<f32>(1.0, 2.0) / 6.28318530718;
    // TODO: hack -v.y
    var uv = vec2<f32>(atan2(v.z, v.x), asin(-v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

fn SampleSky(dir: vec3<f32>) -> vec3<f32> {
    let coords = SampleSphericalMap(dir);
    let dim = textureDimensions(sky);
    let icoords = vec2<i32>(i32(coords.x * f32(dim.x)), i32(coords.y * f32(dim.y)));
    return textureLoad(sky, icoords, 0).rgb;
}

struct Random {
    seed: u32,
};

fn xorshift32(x: ptr<function, u32>) -> u32 {
    var t = *x;
    t ^= t << 13u;
    t ^= t >> 17u;
    t ^= t << 5u;
    *x = t;
    return t;
}

fn Random_Next(this_: ptr<function, Random>) -> f32 {
    let x = &((*this_).seed);
    var seed = (*this_).seed;
    let result = xorshift32(&seed);
    (*this_).seed = seed;
    return f32(result) / f32(0xFFFFFFFFu);
}

fn Random_Init(this_: ptr<function, Random>, gl_GlobalInvocationID: u32) {
    (*this_).seed += gl_GlobalInvocationID + 10u;
    for (var i = 0; i < 8; i++) {
        Random_Next(this_);
    }
}

fn Random_Gaussian(this_: ptr<function, Random>) -> vec2<f32> {
    let polar = vec2<f32>(6.28318530718 * Random_Next(this_), sqrt(-2.0 * log(Random_Next(this_))));
    return vec2<f32>(cos(polar.x) * polar.y, sin(polar.x) * polar.y);
}

fn Random_Disk(this_: ptr<function, Random>) -> vec2<f32> {
    let polar = vec2<f32>(Random_Next(this_), sqrt(Random_Next(this_)));
    return vec2<f32>(cos(6.28318530718 * polar.x) * polar.y, sin(6.28318530718 * polar.x) * polar.y);
}

fn Random_Sphere(this_: ptr<function, Random>) -> vec3<f32> {
    let theta = Random_Next(this_);
    let cosphi = 2.0 * Random_Next(this_) - 1.0;
    let sinphi = sqrt(1.0 - cosphi * cosphi);
    let x = sinphi * cos(6.28318530718 * theta);
    let y = sinphi * sin(6.28318530718 * theta);
    let z = cosphi;
    return vec3<f32>(x, y, z);
}

fn Random_Ball(this_: ptr<function, Random>) -> vec3<f32> {
    return pow(Random_Next(this_), 1.0 / 3.0) * Random_Sphere(this_);
}

fn Random_Lambertian(this_: ptr<function, Random>, normal: vec3<f32>) -> vec3<f32> {
    return normalize(Random_Ball(this_) + normal);
}

struct Ray {
    org: vec3<f32>,
    dir: vec3<f32>,
};

// http://en.wikipedia.org/wiki/Stereographic_projection
fn RayDir(forward: vec3<f32>, upvec: vec3<f32>, screenCoordsp: vec2<f32>, calcFov: f32) -> vec3<f32> {
    var screenCoords = screenCoordsp;
    screenCoords *= -calcFov;
    let len2 = dot(screenCoords, screenCoords);
    let lookvec = vec3<f32>(2.0 * screenCoords.x, 2.0 * screenCoords.y, len2 - 1.0) / -(len2 + 1.0);
    let right = cross(forward, upvec);
    return lookvec.x * right + lookvec.y * upvec + lookvec.z * forward;
}

fn Ray_At(this_: Ray, time: f32) -> vec3<f32> { return this_.org + this_.dir * time; }

fn Ray_Dof(this_: ptr<function, Ray>, focalPlane: f32, rand: ptr<function, Random>) {
    // Normalize because the vectors aren't perpendicular
    let rightUnit = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), (*this_).dir));
    let upUnit = cross((*this_).dir, rightUnit);
    var bloomshift2d: vec2<f32>;
    if Random_Next(rand) < data.bloom_amount {
        bloomshift2d = Random_Gaussian(rand) * data.bloom_size;
    } else {
        bloomshift2d = vec2<f32>(0.0, 0.0);
    }
    bloomshift2d *= focalPlane;
    let bloomshift = bloomshift2d.x * rightUnit + bloomshift2d.y * upUnit;
    var focalPosition = Ray_At(*this_, focalPlane);
    focalPosition += bloomshift;
    let offset = Random_Disk(rand);
    (*this_).dir = normalize((*this_).dir + offset.x * data.dof_amount * rightUnit + offset.y * data.dof_amount * upUnit);
    (*this_).org = focalPosition - (*this_).dir * focalPlane;
}

fn Camera(x: u32, y: u32, width: u32, height: u32, rand: ptr<function, Random>) -> Ray {
// #ifdef NOANTIALIAS
//     vec2 antialias = vec2(0, 0);
// #else
    let antialias = vec2<f32>(Random_Next(rand), Random_Next(rand)) - vec2<f32>(0.5, 0.5);
// #endif
    let screenCoords = vec2<f32>(f32(x) - f32(width) / 2.0, f32(y) - f32(height) / 2.0) + antialias;
    let calcFov = data.fov * 2.0 / f32(width + height);
    let direction = RayDir(data.look.xyz, data.up.xyz, screenCoords, calcFov);
    var result = Ray(data.pos.xyz, direction);
    Ray_Dof(&result, data.focal_distance, rand);
    return result;
}

fn Mandelbulb(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>, power: f32) {
    let zz = *z;
    let r = length(zz.xyz);
    // convert to polar coordinates
    var theta = asin(zz.z / r);
    var phi = atan2(zz.y, zz.x);
    *dz = pow(r, power - 1.0) * power * *dz + 1.0;
    // scale and rotate the point
    let zr = pow(r, power);
    theta = theta * power;
    phi = phi * power;
    // convert back to cartesian coordinates
    *z = zr * vec3(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta));
}

fn Boxfold(z: ptr<function, vec3<f32>>) {
    let limit = vec3<f32>(data.folding_limit);
    *z = clamp(*z, -limit, limit) * 2.0 - *z;
}

fn Spherefold(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>) {
    let factor = data.fixed_radius_2 / clamp(dot(*z, *z), data.min_radius_2, data.fixed_radius_2);
    *dz *= factor;
    *z *= factor;
}

fn TScale(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>) {
    *dz *= abs(data.scale);
    *z *= data.scale;
}

fn TOffset(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>, offset: vec3<f32>) {
    *dz += 1.0;
    *z += offset;
}

fn Rotate(z: vec3<f32>) -> vec3<f32> {
    let axis = normalize(data.plane.xyz);
    let angle = data.rotation;
    return cos(angle) * z + sin(angle) * cross(z, axis) + (1.0 - cos(angle)) * dot(axis, vec3<f32>(angle)) * axis;
}

fn Mandelbox(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>, offset: vec3<f32>, color: ptr<function, u32>) {
    Boxfold(z);
    if dot(*z, *z) < data.min_radius_2 {
        (*color)--;
    } else if dot(*z, *z) < data.fixed_radius_2 {
        (*color)++;
    }
// #ifdef ROTATE
//     z = Rotate(z);
// #endif
    Spherefold(z, dz);
    TScale(z, dz);
    TOffset(z, dz, offset);
}

fn Mandelbulb2(z: ptr<function, vec3<f32>>, dz: ptr<function, f32>, offset: vec3<f32>, color: ptr<function, u32>) {
    Mandelbulb(z, dz, abs(data.scale));
    if *color == 0u {
        *color = 1u << 30u;
    };
    *color = min(*color, u32(dot(*z, *z) * 1000.0));
// #ifdef ROTATE
//     z = Rotate(z);
// #endif
    *z += offset;
}

fn DeSphere(org: vec3<f32>, radius: f32, test: vec3<f32>) -> f32 {
    if radius <= 0.0 {
        return 1E+37;
    }
    return length(test - org) - radius;
}

fn DeMandelbox(offset: vec3<f32>, isNormal: bool, color: ptr<function, u32>) -> f32 {
    var z = vec3<f32>(offset.x, offset.y, offset.z);
    var dz = 1.0f;
    var n = max(data.max_iters, 1u);
    var bail: f32;
    if isNormal {
        bail = data.bailout_normal;
    } else {
        bail = data.bailout;
    }
    loop {
        Mandelbox(&z, &dz, offset, color);
        n = n - 1u;
        if dot(z, z) > bail * bail || n == 0u {
            break;
        }
    }
    return length(z) / dz;
}

fn DeMandelbulb(offset: vec3<f32>, color: ptr<function, u32>) -> f32 {
    var z = vec3<f32>(offset.x, offset.y, offset.z);
    var dz = 1.0f;
    var n = max(data.max_iters, 1u);
    loop {
        Mandelbulb2(&z, &dz, offset, color);
        n = n - 1u;
        if dot(z, z) > 4.0 || n == 0u {
            break;
        }
    }
    let r = length(z);
    return 0.5f * log(r) * r / dz;
}

fn DeFractal(offset: vec3<f32>, isNormal: bool, color: ptr<function, u32>) -> f32 {
// #ifdef MANDELBULB
//         return DeMandelbulb(offset, color);
// #else
    return DeMandelbox(offset, isNormal, color);
// #endif
}

fn Plane(org: vec3<f32>, planedef: vec3<f32>) -> f32 {
    return dot(org, normalize(planedef)) - length(planedef);
}

fn De(offset: vec3<f32>, isNormal: bool) -> f32 {
    var color: u32 = 0u;
    let mbox = DeFractal(offset, isNormal, &color);
// #ifdef PLANE
//     float cut = Plane(offset, plane);
//     return max(mbox, cut);
// #else
    return mbox;
// #endif
}

struct Material {
    color: vec3<f32>,
    normal: vec3<f32>,
    emissive: vec3<f32>,
    gloss: f32,
};

fn GetMaterial(offset: vec3<f32>) -> Material {
    var raw_color_data = 0u;
    let de = DeFractal(offset, true, &raw_color_data);

    let hue = f32(raw_color_data) * data.surface_color_variance + data.surface_color_shift;
    let color = HueToRGB(hue, data.surface_color_saturation, data.surface_color_value);
    var result: Material;
    result.color = color;
    result.gloss = data.surface_color_gloss;
    result.emissive = vec3<f32>(0.0, 0.0, 0.0);

    let delta = max(1e-6f, de * 0.5f); // aprox. 8.3x float epsilon
// #ifdef CUBE_NORMAL
//     float dppp = De(offset + vec3(+delta, + delta, + delta), true);
//     float dppn = De(offset + vec3(+delta, + delta, -delta), true);
//     float dpnp = De(offset + vec3(+delta, -delta, + delta), true);
//     float dpnn = De(offset + vec3(+delta, -delta, -delta), true);
//     float dnpp = De(offset + vec3(-delta, + delta, + delta), true);
//     float dnpn = De(offset + vec3(-delta, + delta, -delta), true);
//     float dnnp = De(offset + vec3(-delta, -delta, + delta), true);
//     float dnnn = De(offset + vec3(-delta, -delta, -delta), true);
//     result.normal = vec3((dppp + dppn + dpnp + dpnn) - (dnpp + dnpn + dnnp + dnnn), (dppp + dppn + dnpp + dnpn) - (dpnp + dpnn + dnnp + dnnn), (dppp + dpnp + dnpp + dnnp) - (dppn + dpnn + dnpn + dnnn));
// #else
    let dnpp = De(offset + vec3<f32>(-delta, delta, delta), true);
    let dpnp = De(offset + vec3<f32>(delta, -delta, delta), true);
    let dppn = De(offset + vec3<f32>(delta, delta, -delta), true);
    let dnnn = De(offset + vec3<f32>(-delta, -delta, -delta), true);
    result.normal = vec3((dppn + dpnp) - (dnpp + dnnn), (dppn + dnpp) - (dpnp + dnnn), (dpnp + dnpp) - (dppn + dnnn));
// #endif
    if dot(result.normal, result.normal) == 0.0f {
        result.normal.x += 1.0; // ensure nonzero
    }
    result.normal = normalize(result.normal);

    return result;
}

fn Cast(ray: Ray, quality: f32, maxDist: f32) -> f32 {
    var distance: f32;
    var totalDistance = 0.0f;
    var i = max(data.max_ray_steps, 1u);
    loop {
        distance = De(Ray_At(ray, totalDistance), false) * data.de_multiplier;
        totalDistance += distance;
        i = i - 1u;
        if totalDistance > maxDist || distance * quality < totalDistance || i == 0u {
            break;
        }
    } 

    // correction step
    if distance * quality <= totalDistance {
        totalDistance -= totalDistance / quality;
        for (var correctStep = 0; correctStep < 4; correctStep++) {
            distance = De(Ray_At(ray, totalDistance), false) * data.de_multiplier;
            totalDistance += distance - totalDistance / quality;
        }
    }
    return totalDistance;
}

fn Trace(rayp: Ray, width: u32, height: u32, rand: ptr<function, Random>) -> vec3<f32> {
    var ray = rayp;
    var rayColor = vec3<f32>(0.0, 0.0, 0.0);
    var reflectionColor = vec3<f32>(1.0, 1.0, 1.0);
    var quality = data.quality_first_ray * (f32(width + height) / (2.0 * data.fov));

    for (var photonIndex = 0u; photonIndex < data.num_ray_bounces; photonIndex++) {
        var fog_dist: f32;
        if data.fog_distance == 0.0 {
            fog_dist = 1E+37;
        } else {
            fog_dist = -log(Random_Next(rand)) * data.fog_distance;
        }
        let max_dist = min(data.max_ray_dist, fog_dist);
        let distance = min(Cast(ray, quality, max_dist), fog_dist);


        if distance >= data.max_ray_dist || (photonIndex + 1u == data.num_ray_bounces && distance >= fog_dist) {
             // went out-of-bounds, or last fog ray didn't hit anything
            let color = SampleSky(ray.dir);
            rayColor += color * reflectionColor;
            break;
        }

        let newPos = Ray_At(ray, min(distance, fog_dist));
        var newDir: vec3<f32>;

        if distance >= fog_dist {
             // hit fog, do fog calculations
            newDir = Random_Sphere(rand);
            reflectionColor *= data.fog_brightness;
        } else {
             // hit surface, do material calculations
            let material = GetMaterial(newPos);
            rayColor += reflectionColor * material.emissive; // ~bling~!
            if Random_Next(rand) < material.gloss {
                newDir = ray.dir;
                 // specular
                if dot(ray.dir, material.normal) < 0.0 {
                    newDir -= 2.0f * dot(ray.dir, material.normal) * material.normal;
                }
                 // material.color = vec3(1.0, 1.0, 1.0);
            } else {
                 // diffuse
                newDir = Random_Lambertian(rand, material.normal);
                quality = data.quality_rest_ray;
                let incident_angle_weakening = dot(material.normal, newDir);
                reflectionColor *= incident_angle_weakening;
            }
            reflectionColor *= material.color;
        }

        ray = Ray(newPos, newDir);

        if dot(reflectionColor, reflectionColor) == 0.0 {
             break;
        }
    }
    return rayColor * data.exposure;
}

fn PreviewTrace(ray: Ray, width: u32, height: u32) -> vec3<f32> {
    let quality = data.quality_first_ray * (f32(width + height) / (2.0 * data.fov));
    let max_dist = min(data.max_ray_dist, data.focal_distance * 10.0);
    let distance = Cast(ray, quality, max_dist);
// #ifdef PREVIEW_NORMAL
//     vec3 org = Ray_At(ray, distance);
//     return abs(GetMaterial(org).normal);
// #else
    let value = distance / max_dist;
    return vec3(value);
// #endif
}

fn GammaTest(x: u32, y: u32, width: u32, height: u32) -> vec3<f32> {
    let centerValue = f32(x) / f32(width);
    let offset = f32((height - y)) / f32(height) * (0.5f - abs(centerValue - 0.5f));
    var result: f32;
    let column_width = 8u;
    if x % (column_width * 2u) < column_width {
        result = centerValue;
    } else if (y & 1u) == 0u {
        result = centerValue + offset;
    } else {
        result = centerValue - offset;
    }
    return vec3(result, result, result);
}

fn GetImg(x: u32, y: u32) -> vec3<f32> {
    return img[y * data.width + x].xyz;
}

fn SetImg(x: u32, y: u32, value: vec3<f32>) {
    img[y * data.width + x] = vec4<f32>(value, 0.0);
}

fn GetRand(x: u32, y: u32, gl_GlobalInvocationID: u32) -> Random {
    let value = randbuf[y * data.width + x];
    var rand = Random(value);
    Random_Init(&rand, gl_GlobalInvocationID);
    return rand;
}

fn SetRand(x: u32, y: u32, value: Random) {
    randbuf[y * data.width + x] = value.seed;
}

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>) {
    let idx = gl_GlobalInvocationID.x;
    let size = data.width * data.height;
    if idx >= size {
        return;
    }
    let x = idx % data.width;
    let y = idx / data.width;

    var newColor: vec3<f32>;
    if data.gamma_test != 0u {
        newColor = GammaTest(x, y, data.width, data.height);
    } else {
        var oldColor: vec3<f32>;
        if data.frame > 0u {
            oldColor = GetImg(x, y);
        } else {
            oldColor = vec3<f32>(0.0);
        }

        var rand = GetRand(x, y, idx);
        let ray = Camera(x, y, data.width, data.height, &rand);
        // #ifdef PREVIEW
        //     vec3 colorComponents = PreviewTrace(ray, width, height);
        // #else
        let colorComponents = Trace(ray, data.width, data.height, &rand);
        newColor = (colorComponents + oldColor * f32(data.frame)) / vec3<f32>(f32(data.frame + 1u));
        SetRand(x, y, rand);
    }
    SetImg(x, y, newColor);
}
