@group(0) @binding(0) 
var tex: texture_2d<f32>;
@group(0) @binding(1) 
var samp: sampler;

@fragment 
fn frag(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, texCoord);
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
