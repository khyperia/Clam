#include <stdio.h>

extern "C" void Main(unsigned char* data,
                     struct MandelboxCfg* cfg_global,
                     unsigned int width,
                     unsigned int height,
                     unsigned int frame);

struct MandelboxCfg
{
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
    float fog_distance;
    float fog_brightness;
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
    float reflect_brightness;
    float surface_color_variance;
    float surface_color_shift;
    float surface_color_saturation;
    float bailout;
    float de_multiplier;
    float max_ray_dist;
    float quality_first_ray;
    float quality_rest_ray;
    int white_clamp;
    int max_iters;
    int max_ray_steps;
    int num_ray_bounces;
};

static struct MandelboxCfg make_default()
{
    struct MandelboxCfg result;
    // s/("\([^ ]*\)", SettingValue::.32(\([^,)]*\).*/result.\1 = \2f;/g
    result.pos_x = 0.0f;
    result.pos_y = 0.0f;
    result.pos_z = 5.0f;
    result.look_x = 0.0f;
    result.look_y = 0.0f;
    result.look_z = -1.0f;
    result.up_x = 0.0f;
    result.up_y = 1.0f;
    result.up_z = 0.0f;
    result.fov = 1.0f;
    result.focal_distance = 3.0f;
    result.scale = -2.0f;
    result.folding_limit = 1.0f;
    result.fixed_radius_2 = 1.0f;
    result.min_radius_2 = 0.125f;
    result.dof_amount = 0.001f;
    result.fog_distance = 10.0f;
    result.fog_brightness = 1.0f;
    result.light_pos_1_x = 3.0f;
    result.light_pos_1_y = 3.5f;
    result.light_pos_1_z = 2.5f;
    result.light_radius_1 = 1.0f;
    result.light_brightness_1_r = 5.0f;
    result.light_brightness_1_g = 5.0f;
    result.light_brightness_1_b = 4.0f;
    result.ambient_brightness_r = 0.8f;
    result.ambient_brightness_g = 0.8f;
    result.ambient_brightness_b = 1.0f;
    result.reflect_brightness = 1.0f;
    result.surface_color_variance = 1.0f;
    result.surface_color_shift = 0.0f;
    result.surface_color_saturation = 1.0f;
    result.bailout = 1024.0f;
    result.de_multiplier = 0.95f;
    result.max_ray_dist = 16.0f;
    result.quality_first_ray = 2.0f;
    result.quality_rest_ray = 64.0f;
    result.white_clamp = 1;
    result.max_iters = 64;
    result.max_ray_steps = 256;
    result.num_ray_bounces = 3;
    return result;
}

int main()
{
    unsigned char data[1000] = {0};
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
