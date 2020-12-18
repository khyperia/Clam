#version 450
// uniform vec4 src_pos_size;
// uniform vec4 dst_pos_size;
layout(location = 0) out vec2 texCoord;
void main()
{
    // float x = (gl_VertexIndex & 1);
    // float y = (gl_VertexIndex & 2) >> 1;
    // float src_x = src_pos_size.x + src_pos_size.z * x;
    // float src_y = src_pos_size.y + src_pos_size.w * y;
    // texCoord.x = src_x;
    // texCoord.y = src_y;
    // float dst_x = dst_pos_size.x + dst_pos_size.z * x;
    // float dst_y = dst_pos_size.y + dst_pos_size.w * y;
    // // flip coordinate space
    // gl_Position = vec4(dst_x * 2 - 1, dst_y * -2 + 1, 0, 1);
    float x = (gl_VertexIndex & 1);
    float y = (gl_VertexIndex & 2) >> 1;
    texCoord.x = x;
    texCoord.y = y;
    // flip coordinate space
    // gl_Position = vec4(x * 2 - 1, y * -2 + 1, 0, 1);
    gl_Position = vec4(x * 2 - 1, y * 2 - 1, 0, 1);
}
