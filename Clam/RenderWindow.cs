using System;
using System.Drawing;
using System.Runtime.InteropServices;
using Cloo;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Clam
{
    class RenderWindow : GameWindow
    {
        private const int InitialHeight = 720;
        private readonly GraphicsInterop _interop;
        private Size _windowSize;
        private double _averageFps;
        private int _lastTitleUpdateSecond;

        private static string _infoMessage;
        private static DateTime _lastInfoMessageUpdate = DateTime.UtcNow;

        public RenderPackage Renderer { get; set; }

        public RenderWindow()
            : base(InitialHeight * 16 / 9, InitialHeight, GraphicsMode.Default, "Clam Render Window",
            GameWindowFlags.Default, DisplayDevice.Default, 0, 0, GraphicsContextFlags.ForwardCompatible)
        {
            var device = ComputePlatform.Platforms[0].Devices[0];
            _interop = new GraphicsInterop(device);
        }

        public ComputeContext ComputeContext { get { return _interop.Context; } }

        public static void DisplayInformation(string information)
        {
            _infoMessage = information;
            _lastInfoMessageUpdate = DateTime.UtcNow;
        }

        protected override void OnResize(EventArgs e)
        {
            _windowSize = ClientSize;
            _interop.OnResize(_windowSize.Width, _windowSize.Height);
            base.OnResize(e);
        }

        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            if (Keyboard[Key.Escape])
                Close();
            base.OnUpdateFrame(e);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            if (!RenderPackage.KernelInUse && Renderer.Kernel != null)
            {
                _interop.Draw(Renderer.Kernel.Render, Renderer.Parameters, _windowSize);
                var frameDependantControls = Renderer.Parameters as IFrameDependantControl;
                if (frameDependantControls != null)
                    frameDependantControls.Frame++;
            }
            _averageFps = (1 / e.Time + _averageFps * 10) / 11;
            var now = DateTime.UtcNow;
            if (now > _lastInfoMessageUpdate + TimeSpan.FromSeconds(5))
            {
                _lastInfoMessageUpdate = now;
                _infoMessage = null;
            }
            if (now.Second != _lastTitleUpdateSecond)
            {
                _lastTitleUpdateSecond = now.Second;
                Title = string.Format("Clam Render Window - {0} fps - {1}", (int)_averageFps, _infoMessage);
            }
            SwapBuffers();
            base.OnRenderFrame(e);
        }

        protected override void OnDisposed(EventArgs e)
        {
            _interop.Dispose();
            Renderer.Dispose();
            base.OnDisposed(e);
        }
    }

    class GraphicsInterop : IDisposable
    {
        [DllImport("opengl32.dll")]
        extern static IntPtr wglGetCurrentDC();

        private readonly ComputeContext _context;
        private readonly int _pub;
        private readonly int _texture;
        private readonly ComputeCommandQueue _queue;
        private ComputeBuffer<Vector4> _openCl;
        private int _width;
        private int _height;

        public GraphicsInterop(ComputeDevice device)
        {
            var glHandle = ((IGraphicsContextInternal)GraphicsContext.CurrentContext).Context.Handle;
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
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.Finish();
            _queue.AcquireGLObjects(new[] { _openCl }, null);
            renderer(_openCl, _queue, parameters, windowSize);
            _queue.ReleaseGLObjects(new[] { _openCl }, null);
            _queue.Finish();
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, _pub);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, _width, _height, PixelFormat.Rgba, PixelType.Float, IntPtr.Zero);
            GL.Begin(BeginMode.Quads);
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
            Context.Dispose();
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
