#version 450
layout(binding = 0) uniform texture2D tex;
layout(binding = 1) uniform sampler samp;
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 out_color;
void main()
{
    // vec4 color1 = texture(sampler2d(tex, samp), texCoord) * scale_offset.x + scale_offset.y;
    // out_color = color1 * tint;
    out_color = texture(sampler2D(tex, samp), texCoord);
}
