using System;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using Cloo;
using Eto.Drawing;
using Eto.Forms;
using System.Collections.Concurrent;

namespace Clam4
{
    internal abstract class Kernel
    {
        public class CompilationException : Exception
        {
        }

        private readonly ComputeKernel _kernel;
        private readonly long _blockSize;

        private readonly ComputeCommandQueue _stream;
        private readonly ComputeEventList _events;

        private Size _screenSize;
        private ComputeBuffer<int> _imageBuffer;

        public static ComputeContext GetContext()
        {
            var p = 0;
            foreach (var platform in ComputePlatform.Platforms)
            {
                var d = 0;
                foreach (var dev in platform.Devices)
                {
                    Console.WriteLine($"{p} {d++} = {dev.Name}");
                }
                p++;
            }
            var device = ComputePlatform.Platforms[0].Devices[1];
            Console.WriteLine(device.Name);
            return new ComputeContext(new[] { device }, new ComputeContextPropertyList(device.Platform), null, IntPtr.Zero);
        }

        protected Kernel(ComputeContext context, string kernel)
        {
            var program = new ComputeProgram(context, kernel);
            //var options = "-Werror -cl-fast-relaxed-math -cl-strict-aliasing";
            var options = "-Werror -cl-fast-relaxed-math";
            program.Build(context.Devices, options, (prog, intptr) =>
            {
                var log = program.GetBuildLog(context.Devices[0]).Trim();
                if (!string.IsNullOrEmpty(log))
                {
                    // We don't have a UI context, so MessageBox fails
                    Console.WriteLine("OpenCL compiler log (maybe error?)");
                    Console.WriteLine(log);
                }
            }, IntPtr.Zero);
            _kernel = program.CreateKernel("Main");
            _blockSize = context.Devices[0].MaxWorkGroupSize;
            _stream = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.OutOfOrderExecution);
            _events = new ComputeEventList();
        }

        protected Kernel(Kernel toClone)
        {
            _kernel = toClone._kernel;
            _blockSize = toClone._blockSize;
            _stream = new ComputeCommandQueue(toClone._stream.Context, toClone._stream.Device, ComputeCommandQueueFlags.OutOfOrderExecution);
            _events = new ComputeEventList();
        }

        public abstract Kernel Clone();

        protected static string LoadResource(string resourceName)
        {
            var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(typeof(Kernel), resourceName);
            stream = stream ?? throw new Exception("Resource not found: " + resourceName);
            using (var reader = new StreamReader(stream))
            {
                return reader.ReadToEnd();
            }
        }

        public abstract SettingsCollection Defaults { get; }
        public abstract IKeyHandler[] KeyHandlers { get; }
        protected abstract void SetArgs(ComputeKernel kernel, ComputeBuffer<int> screen, SettingsCollection settings, Size screenSize, ComputeCommandQueue stream);

        protected ComputeBuffer<T> BufferOfSize<T>(ref ComputeBuffer<T> buffer, Size size, ComputeMemoryFlags flags) where T : struct
        {
            return BufferOfSize(ref buffer, (long)size.Width * size.Height, flags);
        }

        protected ComputeBuffer<T> BufferOfSize<T>(ref ComputeBuffer<T> buffer, long size, ComputeMemoryFlags flags) where T : struct
        {
            if (buffer == null || buffer.Count != size)
            {
                //buffer?.Dispose();
                buffer = new ComputeBuffer<T>(_stream.Context, flags, size);
            }
            return buffer;
        }

        public Size Resize(Size size)
        {
            Console.WriteLine("Resize: " + size.Width + "," + size.Height);
            var oldSize = _screenSize;
            _screenSize = size;
            return oldSize;
        }

        private void RemoveDeadEvents()
        {
            lock (_events)
            {
                while (_events.Count > 0)
                {
                    if (_events[0].Status != ComputeCommandExecutionStatus.Complete)
                    {
                        break;
                    }
                    _events[0].Dispose();
                    _events.RemoveAt(0);
                }
            }
        }

        private static readonly ConcurrentDictionary<IntPtr, TaskCompletionSource<int>> _taskDict = new ConcurrentDictionary<IntPtr, TaskCompletionSource<int>>();

        private static readonly ComputeCommandStatusChanged _onAbort = (obj, status) =>
        {
            TaskCompletionSource<int> value;
            if (!_taskDict.TryRemove(status.Event.Handle.Value, out value))
            {
                throw new Exception("TaskDict did not contain event");
            }
            value.SetException(new Exception(status.Status.ToString()));
        };

        private static readonly ComputeCommandStatusChanged _onComplete = (obj, status) =>
        {
            TaskCompletionSource<int> value;
            if (!_taskDict.TryRemove(status.Event.Handle.Value, out value))
            {
                throw new Exception("TaskDict did not contain event");
            }
            value.SetResult(0);
        };

        private Task ToTask(ComputeEventList events)
        {
            var task = new TaskCompletionSource<int>();
            var finish = _events.Last;
            _taskDict.TryAdd(finish.Handle.Value, task);
            finish.Aborted += _onAbort;
            finish.Completed += _onComplete;
            return task.Task;
        }

        public async Task<ImageData> Run(SettingsCollection settings, bool download)
        {
            RemoveDeadEvents();

            var screenSize = _screenSize;
            var imageBuffer = BufferOfSize(ref _imageBuffer, screenSize, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.AllocateHostPointer);

            SetArgs(_kernel, imageBuffer, settings, screenSize, _stream);

            var launchSize = (long)screenSize.Width * screenSize.Height;
            var localSize = _blockSize;
            var globalSize = (launchSize + localSize - 1) / localSize * localSize;
            _stream.Execute(_kernel, new[] { 0L }, new[] { globalSize }, new[] { localSize }, _events);

            if (!download)
            {
                await ToTask(_events);
                return new ImageData();
            }

            var mappedImageBuffer = _stream.Map(imageBuffer, false, ComputeMemoryMappingFlags.Read, 0, launchSize, _events);
            //_stream.ReadFromBuffer(imageBuffer, ref buffer, false, _events);
            await ToTask(_events);
            var result = new ImageData(screenSize.Width, screenSize.Height, (int)launchSize);
            Marshal.Copy(mappedImageBuffer, result.Buffer, 0, result.Buffer.Length);
            _stream.Unmap(imageBuffer, ref mappedImageBuffer, _events);
            return result;
        }
    }

    internal sealed class Mandelbrot : Kernel
    {
        public Mandelbrot(ComputeContext context) : base(context, LoadResource("mandelbrot.cl"))
        {
        }

        private Mandelbrot(Mandelbrot toClone) : base(toClone)
        {
        }

        public override Kernel Clone() => new Mandelbrot(this);

        protected override void SetArgs(ComputeKernel kernel, ComputeBuffer<int> screen, SettingsCollection settings, Size screenSize, ComputeCommandQueue stream)
        {
            var posX = (float)(double)settings["posX"];
            var posY = (float)(double)settings["posY"];
            var zoom = (float)(double)settings["zoom"];

            int arg = 0;
            kernel.SetMemoryArgument(arg++, screen);
            kernel.SetValueArgument(arg++, screenSize.Width / -2);
            kernel.SetValueArgument(arg++, screenSize.Height / -2);
            kernel.SetValueArgument(arg++, screenSize.Width);
            kernel.SetValueArgument(arg++, screenSize.Height);
            kernel.SetValueArgument(arg++, posX);
            kernel.SetValueArgument(arg++, posY);
            kernel.SetValueArgument(arg++, zoom);
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
        struct Float4
        {
            float x;
            float y;
            float z;
            float w;
        }
        struct Uint2
        {
            int x;
            int y;
        }
        private ComputeBuffer<MandelboxCfg> _cfg;
        private ComputeBuffer<Float4> _scratchBuf;
        private ComputeBuffer<Uint2> _randBuf;
        private MandelboxCfg _oldConfig;
        private bool _needsRenorm;
        private int _frame;

        public Mandelbox(ComputeContext context) : base(context, LoadResource("mandelbox.cl"))
        {
        }

        private Mandelbox(Mandelbox toClone) : base(toClone)
        {
        }

        public override Kernel Clone() => new Mandelbox(this);

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
                    ["LightBrightnessR"] = 9.0f,
                    ["LightBrightnessG"] = 2.0f,
                    ["LightBrightnessB"] = 2.0f,
                    ["AmbientBrightnessR"] = 0.55f,
                    ["AmbientBrightnessG"] = 0.9f,
                    ["AmbientBrightnessB"] = 0.9f,
                    ["ReflectBrightness"] = 0.95f,
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

        protected override void SetArgs(ComputeKernel kernel, ComputeBuffer<int> screen, SettingsCollection settings, Size screenSize, ComputeCommandQueue stream)
        {
            var oldScratchBuf = _scratchBuf;
            var scratchBuf = BufferOfSize(ref _scratchBuf, screenSize, ComputeMemoryFlags.ReadWrite);
            var randBuf = BufferOfSize(ref _randBuf, screenSize, ComputeMemoryFlags.ReadWrite);

            if (_needsRenorm)
            {
                _keyCamera.NormalizeLookUp(settings);
                _needsRenorm = false;
            }
            var config = GetCfg(settings);
            if (!_oldConfig.Equals(config) || oldScratchBuf != scratchBuf)
            {
                _oldConfig = config;
                if (_cfg == null)
                {
                    _cfg = new ComputeBuffer<MandelboxCfg>(kernel.Context, ComputeMemoryFlags.ReadOnly, 1);
                }
                stream.WriteToBuffer(new[] { config }, _cfg, true, null);
                _frame = 0;
            }

            int arg = 0;
            kernel.SetMemoryArgument(arg++, screen);
            kernel.SetMemoryArgument(arg++, _cfg);
            kernel.SetMemoryArgument(arg++, scratchBuf);
            kernel.SetMemoryArgument(arg++, randBuf);
            kernel.SetValueArgument(arg++, screenSize.Width / -2);
            kernel.SetValueArgument(arg++, screenSize.Height / -2);
            kernel.SetValueArgument(arg++, screenSize.Width);
            kernel.SetValueArgument(arg++, screenSize.Height);
            kernel.SetValueArgument(arg++, _frame++);
        }

        private MandelboxCfg GetCfg(SettingsCollection settings)
        {
            object cfg = new MandelboxCfg();
            foreach (var field in cfg.GetType().GetFields())
            {
                var name = field.Name;
                var value = settings[name];
                field.SetValue(cfg, value);
            }
            return (MandelboxCfg)cfg;
        }

        struct MandelboxCfg
        {
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
            public float LightBrightnessR;
            public float LightBrightnessG;
            public float LightBrightnessB;
            public float AmbientBrightnessR;
            public float AmbientBrightnessG;
            public float AmbientBrightnessB;
            public float ReflectBrightness;
            public float Bailout;
            public float DeMultiplier;
            public float MaxRayDist;
            public float QualityFirstRay;
            public float QualityRestRay;
            public int WhiteClamp;
            public int MaxIters;
            public int RandSeedInitSteps;
            public int MaxRaySteps;
            public int NumRayBounces;
        };
    }
}
