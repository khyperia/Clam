#include <stdio.h>

extern "C" void Main(
    unsigned char* data,
    struct MandelboxCfg* cfg_global,
    unsigned int width,
    unsigned int height,
    unsigned int frame);

struct MandelboxCfg {
    float pos_x;
    float pos_y;
    float pos_z;
    float look_x;
    float look_y;
    float look_z;
    float up_x;
    float up_y;
    float up_z;
    float fov;
    float focal_distance;
    float scale;
    float folding_limit;
    float fixed_radius_2;
    float min_radius_2;
    float dof_amount;
    float light_pos_1_x;
    float light_pos_1_y;
    float light_pos_1_z;
    float light_radius_1;
    float light_brightness_1_r;
    float light_brightness_1_g;
    float light_brightness_1_b;
    float ambient_brightness_r;
    float ambient_brightness_g;
    float ambient_brightness_b;
    float fog_distance;
    float fog_scatter;
    float reflect_brightness;
    float bailout;
    float de_multiplier;
    float max_ray_dist;
    float quality_first_ray;
    float quality_rest_ray;
    int white_clamp;
    int max_iters;
    int max_ray_steps;
};

struct MandelboxCfg make_default() {
    struct MandelboxCfg result;
    // s/("\([^ ]*\)", \([^,]*\).*/result.\1 = \2;/g
    result.pos_x = 0.0;
    result.pos_y = 0.0;
    result.pos_z = 5.0;
    result.look_x = 0.0;
    result.look_y = 0.0;
    result.look_z = -1.0;
    result.up_x = 0.0;
    result.up_y = 1.0;
    result.up_z = 0.0;
    result.fov = 1.0;
    result.focal_distance = 3.0;
    result.scale = -2.0;
    result.folding_limit = 1.0;
    result.fixed_radius_2 = 1.0;
    result.min_radius_2 = 0.125;
    result.dof_amount = 0.001;
    result.light_pos_1_x = 3.0;
    result.light_pos_1_y = 3.5;
    result.light_pos_1_z = 2.5;
    result.light_radius_1 = 1.0;
    result.light_brightness_1_r = 5.0;
    result.light_brightness_1_g = 5.0;
    result.light_brightness_1_b = 4.0;
    result.ambient_brightness_r = 0.8;
    result.ambient_brightness_g = 0.8;
    result.ambient_brightness_b = 1.0;
    result.fog_distance = 1000.0;
    result.fog_scatter = 1.0;
    result.reflect_brightness = 1.0;
    result.bailout = 1024.0;
    result.de_multiplier = 0.95;
    result.max_ray_dist = 16.0;
    result.quality_first_ray = 2.0;
    result.quality_rest_ray = 64.0;
    result.white_clamp = 1.0;
    result.max_iters = 64.0;
    result.max_ray_steps = 256.0;
    return result;
}

int main()
{
    unsigned char data[1000];
    struct MandelboxCfg cfg = make_default();
    Main(data, &cfg, 1, 1, 0);
    for (int i = 0; i < 50; i++)
    {
        printf("%d\n", data[i]);
    }
    printf("hello, world\n");
    return 0;
}

unsigned int get_global_id(unsigned int);
unsigned int get_global_id(unsigned int)
{
    return 0;
}
