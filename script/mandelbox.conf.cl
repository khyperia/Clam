#define Gauss(a, c, w, x) ((a) * exp(-(((x) - (c)) * ((x) - (c))) / (float)(2 * (w) * (w))))

#define Transform4Dx 1,0,0,0
#define Transform4Dy 0,1,0,0
#define Transform4Dz 0,0,1,0
#define Scale 2.43
#define FoldingLimit 1.0
#define FixedRadius2 1.0
#define MinRadius2 0.125
#define ColorSharpness 1.0
#define Saturation 0.6
#define HueVariance 0.01
#define Reflectivity 1.0
#define ColorBias 0.0,0.0,0.0
#define DofAmount(hue) (hue * 0.01)
#define FovAbberation 0.01
#define SpecularHighlight(angle) Gauss(5, 0, 0.2, angle)
#define LightBrightness(hue) Gauss(5, 0.5, 10.0, hue)
#define AmbientBrightness(hue) Gauss(16, 0.5, 10.0, hue)
#define LightPos 10
#define LightSize 0.2
#define FogDensity(hue) (hue * 0.000001)
#define FogColor(hue) Gauss(1, 0.5, 10.0, hue)
#define GlassPos 2.5,1,1
#define GlassSize 0.75
#define GlassDensity 1.2

#define MaxIters 32
#define Bailout 128
#define DeMultiplier 0.95
#define RandSeedInitSteps 128
#define MaxRayDist 64
#define MaxRaySteps 256
#define DirectLightingMaxSteps 64
#define NumRayBounces 3
#define QualityFirstRay 4
#define QualityRestRay 64
#define DirectLightProbability 0.4
