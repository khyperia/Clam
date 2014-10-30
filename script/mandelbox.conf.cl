#define Gauss(a, c, w, x) ((a) * exp(-(((x) - (c)) * ((x) - (c))) / (2 * (w) * (w))))

#define Transform4Dx 1,0,0,0
#define Transform4Dy 0,1,0,0
#define Transform4Dz 0,0,1,0
#define Scale -1.5
#define FoldingLimit 1.0
#define FixedRadius2 1.0
#define MinRadius2 0.125
#define Saturation 0.8
#define HueVariance 0.01
#define Reflectivity 1.0
#define ColorBias 0.0,0.0,0.0
#define DofAmount(hue) (hue * 0.01)
#define FovAbberation 0.03
#define SpecularSize 0.4
#define SpecularBrightness 3
#define LightBrightness(hue) Gauss(10000, 0.1, 0.7, hue)
#define AmbientBrightness(hue) Gauss(1, 0.9, 2, hue)
#define LightPos 6
#define LightSize 0.2
#define FogDensity(hue) 0.1
#define FogColor(hue) Gauss(1, 0.75, 0.5, hue)

#define MaxIters 32
#define Bailout 128
#define DeMultiplier 0.95
#define RandSeedInitSteps 128
#define MaxRayDist 64
#define MaxRaySteps 256
#define DirectLightingMaxSteps 64
#define NumRayBounces 2
#define QualityFirstRay 1
#define QualityRestRay 64
#define DirectLightProbability 0.4
