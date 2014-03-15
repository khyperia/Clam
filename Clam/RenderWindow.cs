using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading;
using Cloo;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Clam
{
    public class RenderWindow : GLControl
    {
        private readonly Action<string> _setStatusBar;
        private readonly GraphicsInterop _interop;
        private Size _windowSize;
        private double _averageFps;
        private int _lastTitleUpdateSecond;
        private RenderPackage _renderer;
        private DateTime _lastUpdate;

        private static string _infoMessage;
        private static DateTime _lastInfoMessageUpdate = DateTime.UtcNow;

        public RenderWindow(ComputeDevice device, Action<string> setStatusBar)
            : base(GraphicsMode.Default, 0, 0, GraphicsContextFlags.ForwardCompatible)
        {
            _setStatusBar = setStatusBar;
            _interop = new GraphicsInterop(Context, device);
            _lastUpdate = DateTime.UtcNow;
            ThreadPool.QueueUserWorkItem(DoTimerTick);
        }

        public ComputeContext ComputeContext { get { return _interop.Context; } }

        public RenderPackage Renderer
        {
            get { return _renderer; }
            set
            {
                if (_renderer.Kernel != null && _renderer.Kernel != value.Kernel)
                    _renderer.Kernel.Dispose();
                _renderer = value;
            }
        }

        protected override void OnResize(EventArgs e)
        {
            _windowSize = ClientSize;
            _interop.OnResize(_windowSize.Width, _windowSize.Height);
            var fdc = _renderer.Parameters as IFrameDependantControl;
            if (fdc != null)
                fdc.Frame = 0;
            base.OnResize(e);
        }

        public void Invoke(Action action)
        {
            base.Invoke(action);
        }

        public void DisplayInformation(string information)
        {
            _infoMessage = information;
        }

        private void DoTimerTick(object state)
        {
            var timeToSleep = _lastUpdate + TimeSpan.FromMilliseconds(1000 / 60) - DateTime.UtcNow;
            if (timeToSleep.TotalSeconds > 0)
                Thread.Sleep(timeToSleep);
            Invoke(OnTimerTick);
        }

        private void OnTimerTick()
        {
            var paramSet = Renderer.Parameters as IUpdateableParameterSet;
            if (paramSet != null)
                paramSet.Update((DateTime.UtcNow - _lastUpdate).TotalSeconds, Focused);
            _averageFps = (1 / (DateTime.UtcNow - _lastUpdate).TotalSeconds + _averageFps * 10) / 11;
            _lastUpdate = DateTime.UtcNow;
            var now = DateTime.UtcNow;
            if (now > _lastInfoMessageUpdate + TimeSpan.FromSeconds(5))
            {
                _lastInfoMessageUpdate = now;
                _infoMessage = null;
            }
            if (now.Second != _lastTitleUpdateSecond)
            {
                _lastTitleUpdateSecond = now.Second;
                _setStatusBar(string.Format("{0} fps - {1}", (int)_averageFps, _infoMessage));
            }

            if (!RenderPackage.KernelInUse && Renderer.Kernel != null)
            {
                _interop.Draw(Renderer.Kernel.Render, Renderer.Parameters, _windowSize);
                var frameDependantControls = Renderer.Parameters as IFrameDependantControl;
                if (frameDependantControls != null)
                    frameDependantControls.Frame++;
            }

            SwapBuffers();

            ThreadPool.QueueUserWorkItem(DoTimerTick);
        }

        protected override void Dispose(bool manual)
        {
            if (manual)
            {
                _interop.Dispose();
                Renderer.Dispose();
            }
            base.Dispose(manual);
        }
    }

    class GraphicsInterop : IDisposable
    {
        [DllImport("opengl32.dll")]
        private extern static IntPtr wglGetCurrentDC();

        private readonly ComputeContext _context;
        private readonly int _pub;
        private readonly int _texture;
        private readonly ComputeCommandQueue _queue;
        private ComputeBuffer<Vector4> _openCl;
        private int _width;
        private int _height;
        private volatile bool _disposed;

        public GraphicsInterop(IGraphicsContext context, ComputeDevice device)
        {
            var currentContext = (IGraphicsContextInternal)context;
            var glHandle = currentContext.Context.Handle;
            var wglHandle = wglGetCurrentDC();
            var p1 = new ComputeContextProperty(ComputeContextPropertyName.Platform, device.Platform.Handle.Value);
            var p2 = new ComputeContextProperty(ComputeContextPropertyName.CL_GL_CONTEXT_KHR, glHandle);
            var p3 = new ComputeContextProperty(ComputeContextPropertyName.CL_WGL_HDC_KHR, wglHandle);
            var cpl = new ComputeContextPropertyList(new[] { p1, p2, p3 });
            _context = new ComputeContext(ComputeDeviceTypes.Gpu, cpl, null, IntPtr.Zero);
            _queue = new ComputeCommandQueue(Context, device, ComputeCommandQueueFlags.None);
            GL.ClearColor(0f, 0f, 1f, 1f);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
            GL.Ortho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            GL.GenBuffers(1, out _pub);
            GL.Enable(EnableCap.Texture2D);
            _texture = GL.GenTexture();
        }

        public void OnResize(int width, int height)
        {
            if (_disposed)
                throw new ObjectDisposedException(ToString());
            _width = width;
            _height = height;
            GL.Viewport(0, 0, width, height);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, _pub);
            GL.BufferData(BufferTarget.PixelUnpackBuffer, new IntPtr(width * height * sizeof(float) * 4), IntPtr.Zero, BufferUsageHint.DynamicCopy);
            if (_openCl != null)
                _openCl.Dispose();
            _openCl = ComputeBuffer<Vector4>.CreateFromGLBuffer<Vector4>(_queue.Context, ComputeMemoryFlags.WriteOnly, _pub);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, width, height, 0, PixelFormat.Rgba, PixelType.Float, IntPtr.Zero);
            const int glLinear = 9729;
            GL.TexParameterI(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, new[] { glLinear });
            GL.TexParameterI(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, new[] { glLinear });
        }

        public void Draw(Action<ComputeBuffer<Vector4>, ComputeCommandQueue, IParameterSet, Size> renderer, IParameterSet parameters, Size windowSize)
        {
            if (_disposed)
                throw new ObjectDisposedException(ToString());
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.Finish();
            _queue.AcquireGLObjects(new[] { _openCl }, null);
            renderer(_openCl, _queue, parameters, windowSize);
            _queue.ReleaseGLObjects(new[] { _openCl }, null);
            _queue.Finish();
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, _pub);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, _width, _height, PixelFormat.Rgba, PixelType.Float, IntPtr.Zero);
            GL.Begin(PrimitiveType.Quads);
            GL.TexCoord2(0f, 1f);
            GL.Vertex3(0f, 0f, 0f);
            GL.TexCoord2(0f, 0f);
            GL.Vertex3(0f, 1f, 0f);
            GL.TexCoord2(1f, 0f);
            GL.Vertex3(1f, 1f, 0f);
            GL.TexCoord2(1f, 1f);
            GL.Vertex3(1f, 0f, 0f);
            GL.End();
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _context.Dispose();
            _queue.Dispose();
            _openCl.Dispose();
            var tmpPub = _pub;
            GL.DeleteBuffers(1, ref tmpPub);
            GL.DeleteTexture(_texture);
        }

        public ComputeContext Context
        {
            get { return _context; }
        }
    }
}
