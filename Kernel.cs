using System;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Clam4
{
    internal abstract class Kernel
    {
        public class CompilationException : Exception
        {
        }

        private readonly CudaKernel _kernel;
        private readonly int _blockSize;

        private readonly CudaStream _stream;

        private Size _screenSize;
        private CudaPageLockedHostMemory_int _cpuMemory;
        private CudaDeviceVariable<int> _imageBuffer;

        protected Kernel(CudaContext context, CodeObject kernel)
        {
            var ptx = kernel.PTX;
            if (ptx == null || ptx.Length == 0)
            {
                throw new CompilationException();
            }
            _kernel = context.LoadKernelPTX(ptx, "Main");
            _blockSize = _kernel.GetOccupancyMaxPotentialBlockSize().blockSize;
            _stream = new CudaStream();
        }

        public abstract SettingsCollection Defaults { get; }
        public abstract IKeyHandler[] KeyHandlers { get; }
        protected abstract object[] GetArgs(CudaKernel kernel, SettingsCollection settings, Size screenSize, CudaStream stream);

        protected static CudaDeviceVariable<T> BufferOfSize<T>(ref CudaDeviceVariable<T> buffer, Size size) where T : struct
        {
            return BufferOfSize(ref buffer, (SizeT)size.Width * (SizeT)size.Height);
        }

        protected static CudaDeviceVariable<T> BufferOfSize<T>(ref CudaDeviceVariable<T> buffer, SizeT size) where T : struct
        {
            if (buffer == null || buffer.Size != size)
            {
                buffer?.Dispose();
                buffer = new CudaDeviceVariable<T>(size);
            }
            return buffer;
        }

        public virtual void Resize(Size size)
        {
            var imageBuffer = BufferOfSize(ref _imageBuffer, size);

            if (_cpuMemory == null || _cpuMemory.Size != imageBuffer.Size)
            {
                _cpuMemory?.Dispose();
                _cpuMemory = new CudaPageLockedHostMemory_int(imageBuffer.Size);
            }

            _screenSize = size;
        }

        public Task<ImageData> Run(SettingsCollection settings)
        {
            var screenSize = _screenSize;
            var imageBuffer = _imageBuffer;
            var cpuMemory = _cpuMemory;
            var args = GetArgs(_kernel, settings, screenSize, _stream);
            // assume args[0] is image buffer
            if (args[0] != null)
            {
                throw new Exception("args[0] not null");
            }
            args[0] = imageBuffer.DevicePointer;

            var launchSize = screenSize.Width * screenSize.Height;
            _kernel.BlockDimensions = new dim3(_blockSize);
            _kernel.GridDimensions = new dim3((launchSize + _blockSize - 1) / _blockSize);
            _kernel.RunAsync(_stream.Stream, args);

            cpuMemory.AsyncCopyFromDevice(imageBuffer, _stream.Stream);
            var task = new TaskCompletionSource<ImageData>();
            CUstreamCallback func = CallbackData.Go;
            GCHandle handle = GCHandle.Alloc(new CallbackData(func, cpuMemory, task, screenSize));
            _stream.AddCallback(func, GCHandle.ToIntPtr(handle), CUStreamAddCallbackFlags.None);
            return task.Task;
        }

        private class CallbackData
        {
            private readonly CUstreamCallback _streamCallback;
            private readonly CudaPageLockedHostMemory_int _cpuMemory;
            private readonly TaskCompletionSource<ImageData> _task;
            private readonly Size _size;

            public CallbackData(CUstreamCallback streamCallback, CudaPageLockedHostMemory_int cpuMemory, TaskCompletionSource<ImageData> task, Size size)
            {
                _streamCallback = streamCallback;
                _cpuMemory = cpuMemory;
                _task = task;
                _size = size;
            }

            private void Run(CUResult status)
            {
                if (status != CUResult.Success)
                {
                    throw new CudaException(status);
                }
                int size = _cpuMemory.Size;
                var imageData = new ImageData(_size.Width, _size.Height, size);
                Marshal.Copy(_cpuMemory.PinnedHostPointer, imageData.Buffer, 0, size);
                _task.SetResult(imageData);
            }

            public static void Go(CUstream stream, CUResult status, IntPtr userdata)
            {
                var handle = GCHandle.FromIntPtr(userdata);
                try
                {
                    ((CallbackData)handle.Target).Run(status);
                }
                catch (Exception e)
                {
                    Console.WriteLine("Exception in kernel callback handler! - " + e);
                }
                finally
                {
                    handle.Free();
                }
            }
        }
    }

    internal sealed class Mandelbrot : Kernel
    {
        public Mandelbrot(CudaContext context) : base(context, new CodeObject("mandelbrot.cu"))
        {
        }

        protected override object[] GetArgs(CudaKernel kernel, SettingsCollection settings, Size screenSize, CudaStream stream)
        {
            var posX = (float)(double)settings["posX"];
            var posY = (float)(double)settings["posY"];
            var zoom = (float)(double)settings["zoom"];

            return new object[]
            {
                null,
                screenSize.Width / -2,
                screenSize.Height / -2,
                screenSize.Width,
                screenSize.Height,
                posX,
                posY,
                zoom
            };
        }

        public override SettingsCollection Defaults =>
            new SettingsCollection()
            {
                ["posX"] = 0.0,
                ["posY"] = 0.0,
                ["zoom"] = 1.0,
            };

        public override IKeyHandler[] KeyHandlers =>
            new IKeyHandler[]
            {
                new KeyPan2d(Keys.W, Keys.S, Keys.A, Keys.D, Keys.R, Keys.F, "posX", "posY", "zoom")
            };
    }

    internal sealed class Mandelbox : Kernel
    {
        private CudaDeviceVariable<MandelboxCfg> _cfg;
        private CudaPageLockedHostMemory<MandelboxCfg> _cfgCpu;
        private CudaDeviceVariable<float4> _scratchBuf;
        private CudaDeviceVariable<uint2> _randBuf;
        private MandelboxCfg _oldConfig;
        private bool _needsRenorm;
        private int _frame;

        public Mandelbox(CudaContext context) : base(context, new CodeObject("mandelbox.cu"))
        {
        }

        public override SettingsCollection Defaults
        {
            get
            {
                var result = new SettingsCollection()
                {
                    ["posX"] = 0.0f,
                    ["posY"] = 0.0f,
                    ["posZ"] = 5.0f,
                    ["lookX"] = 0.0f,
                    ["lookY"] = 0.0f,
                    ["lookZ"] = -1.0f,
                    ["upX"] = 0.0f,
                    ["upY"] = 1.0f,
                    ["upZ"] = 0.0f,
                    ["fov"] = 1.0f,
                    ["focalDistance"] = 3.0f,
                    ["Scale"] = -2.0f,
                    ["FoldingLimit"] = 1.0f,
                    ["FixedRadius2"] = 1.0f,
                    ["MinRadius2"] = 0.125f,
                    ["DofAmount"] = 0.01f,
                    ["LightPosX"] = 3.0f,
                    ["LightPosY"] = 4.0f,
                    ["LightPosZ"] = 3.0f,
                    ["WhiteClamp"] = 0,
                    ["LightBrightnessHue"] = 0.05f,
                    ["LightBrightnessSat"] = 0.7f,
                    ["LightBrightnessVal"] = 8.0f,
                    ["AmbientBrightnessHue"] = 0.55f,
                    ["AmbientBrightnessSat"] = 0.7f,
                    ["AmbientBrightnessVal"] = 0.4f,
                    ["ReflectBrightnessHue"] = 0.5f,
                    ["ReflectBrightnessSat"] = 0.0f,
                    ["ReflectBrightnessVal"] = 0.95f,
                    ["MaxIters"] = 32,
                    ["Bailout"] = 1024.0f,
                    ["DeMultiplier"] = 0.95f,
                    ["RandSeedInitSteps"] = 8,
                    ["MaxRayDist"] = 16.0f,
                    ["MaxRaySteps"] = 64,
                    ["NumRayBounces"] = 3,
                    ["QualityFirstRay"] = 4.0f,
                    ["QualityRestRay"] = 128.0f,
                };
                result.OnChange += (key, value) =>
                {
                    _needsRenorm = true;
                };
                return result;
            }
        }

        private readonly KeyCamera3d _keyCamera = new KeyCamera3d(
                    "posX",
                    "posY",
                    "posZ",
                    "lookX",
                    "lookY",
                    "lookZ",
                    "upX",
                    "upY",
                    "upZ",
                    "focalDistance",
                    "fov");

        public override IKeyHandler[] KeyHandlers =>
            new IKeyHandler[] {
                _keyCamera,
                new KeyScale(Keys.R, Keys.F, "focalDistance", -0.5),
                new KeyScale(Keys.N, Keys.M, "fov", -0.5),
            };

        public override void Resize(Size size)
        {
            base.Resize(size);
            BufferOfSize(ref _randBuf, size);
            BufferOfSize(ref _scratchBuf, size);
        }

        protected override object[] GetArgs(CudaKernel kernel, SettingsCollection settings, Size screenSize, CudaStream stream)
        {
            if (_needsRenorm)
            {
                _keyCamera.NormalizeLookUp(settings);
                _needsRenorm = false;
            }
            var config = GetCfg(settings);
            if (!_oldConfig.Equals(config))
            {
                _oldConfig = config;
                if (_cfg == null)
                {
                    _cfg = new CudaDeviceVariable<MandelboxCfg>(kernel, "CfgArr");
                    _cfgCpu = new CudaPageLockedHostMemory<MandelboxCfg>(1);
                }
                Marshal.StructureToPtr(_oldConfig, _cfgCpu.PinnedHostPointer, false);
                _cfgCpu.AsyncCopyToDevice(_cfg, stream.Stream);
                _frame = 0;
            }

            return new object[]
            {
                null,
                screenSize.Width / -2,
                screenSize.Height / -2,
                screenSize.Width,
                screenSize.Height,
                _frame++,
            };
        }

        private MandelboxCfg GetCfg(SettingsCollection settings)
        {
            object cfg = new MandelboxCfg();
            foreach (var field in cfg.GetType().GetFields())
            {
                var name = field.Name;
                object value;
                if (name == "scratch")
                {
                    value = _scratchBuf.DevicePointer;
                }
                else if (name == "randbuf")
                {
                    value = _randBuf.DevicePointer;
                }
                else
                {
                    value = settings[name];
                }
                field.SetValue(cfg, value);
            }
            return (MandelboxCfg)cfg;
        }

        struct MandelboxCfg
        {
            public CUdeviceptr scratch;
            public CUdeviceptr randbuf;
            public float posX;
            public float posY;
            public float posZ;
            public float lookX;
            public float lookY;
            public float lookZ;
            public float upX;
            public float upY;
            public float upZ;
            public float fov;
            public float focalDistance;
            public float Scale;
            public float FoldingLimit;
            public float FixedRadius2;
            public float MinRadius2;
            public float DofAmount;
            public float LightPosX;
            public float LightPosY;
            public float LightPosZ;
            public int WhiteClamp;
            public float LightBrightnessHue;
            public float LightBrightnessSat;
            public float LightBrightnessVal;
            public float AmbientBrightnessHue;
            public float AmbientBrightnessSat;
            public float AmbientBrightnessVal;
            public float ReflectBrightnessHue;
            public float ReflectBrightnessSat;
            public float ReflectBrightnessVal;
            public int MaxIters;
            public float Bailout;
            public float DeMultiplier;
            public int RandSeedInitSteps;
            public float MaxRayDist;
            public int MaxRaySteps;
            public int NumRayBounces;
            public float QualityFirstRay;
            public float QualityRestRay;
        };
    }
}
