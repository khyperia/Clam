struct Uniforms {
    width: u32,
    height: u32,
    output_srgb: u32,
    dummy: u32,
}

@group(0) @binding(0) 
var<storage> tex: array<vec4<f32>>;
@group(0) @binding(1) 
var<uniform> unis: Uniforms;

fn LinearToSrgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        return 12.92 * value;
    } else {
        return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
    }
}

@fragment 
fn frag(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let x = u32(texCoord.x * f32(unis.width));
    let y = u32(texCoord.y * f32(unis.height));
    var value = tex[y * unis.width + x];
    if unis.output_srgb != 0u {
        value.x = LinearToSrgb(value.x);
        value.y = LinearToSrgb(value.y);
        value.z = LinearToSrgb(value.z);
    }
    return value;
}

struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) member: vec4<f32>,
}

@vertex 
fn vert(@builtin(vertex_index) gl_VertexIndex: u32) -> VertexOutput {
    var x = f32(gl_VertexIndex & 1u);
    var y = f32((gl_VertexIndex & 2u) >> 1u);
    var texCoord = vec2<f32>(x, y);
    var gl_Position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return VertexOutput(texCoord, gl_Position);
}
