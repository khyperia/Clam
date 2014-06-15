using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using Clam.NGif;
using Cloo;
using JetBrains.Annotations;
using OpenTK;
using Size = System.Drawing.Size;

namespace Clam
{
    public struct RenderPackage : IDisposable
    {
        private readonly RenderKernel _kernel;
        private readonly IParameterSet _parameters;
        private static volatile int _kernelInUse;

        public RenderPackage(RenderKernel kernel, IParameterSet parameters)
        {
            _kernel = kernel;
            _parameters = parameters;
        }

        [CanBeNull]
        public RenderKernel Kernel
        {
            get { return _kernel; }
        }

        [CanBeNull]
        public IParameterSet Parameters
        {
            get { return _parameters; }
        }

        public static bool KernelInUse
        {
            get { return _kernelInUse != 0; }
        }

        private static readonly Dictionary<string, Func<IParameterSet>> ControlBindings = new Dictionary<string, Func<IParameterSet>>
        {
            {"Raytracer", () => new KeyboardRaytracerControl()},
            {"RaytracerExtended", () => new KeyboardRaytracerControlExtended()},
            {"2D", () => new Keyboard2DControl()},
            {"2DFrame", () => new Keyboard2DFrame()},
        };

        public static Dictionary<string, Func<IParameterSet>> ControlBindingNames
        {
            get { return ControlBindings; }
        }

        public static RenderPackage? LoadFromXml(ComputeContext computeContext, KernelXmlFile kernelXml, IParameterSet oldParameterSetCache)
        {
            var kernel = RenderKernel.Create(computeContext, kernelXml.Files.Select(File.ReadAllText).ToArray());
            if (kernel == null)
                return null;
            var controls = kernelXml.ControlsFunc();
            if (oldParameterSetCache != null && controls.GetType() == oldParameterSetCache.GetType())
                controls = oldParameterSetCache;
            return new RenderPackage(kernel, controls);
        }

        private const double ScreenshotAspectRatio = 16.0 / 9.0;

        [CanBeNull]
        public Bitmap Screenshot(int screenshotHeight, int slowRenderPower, Action<string> displayInformation)
        {
            displayInformation("Rendering screenshot");
            var ccontext = _kernel.ComputeContext;
            _kernelInUse++;

            var screenshotWidth = (int)(screenshotHeight * ScreenshotAspectRatio);
            Bitmap bmp;
            try
            {
                bmp = new Bitmap(screenshotWidth, screenshotHeight, PixelFormat.Format24bppRgb);
            }
            catch (ArgumentException)
            {
                MessageBox.Show("Image size too big", "Error");
                return null;
            }
            var nancount = 0;
            var bmpData = bmp.LockBits(new Rectangle(0, 0, screenshotWidth, screenshotHeight), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            var scan0 = bmpData.Scan0.ToInt64();
            var queue = new ComputeCommandQueue(ccontext, ccontext.Devices[0], ComputeCommandQueueFlags.None);
            var localSize = _kernel.Threadsize(queue);
            for (var i = 0; i < localSize.Length; i++)
                localSize[i] *= slowRenderPower;
            var computeBuffer = new ComputeBuffer<Vector4>(ccontext, ComputeMemoryFlags.ReadWrite, localSize[0] * localSize[1]);

            const int numFrames = 200;
            var frameDependantControls = _parameters as IFrameDependantControl;
            var framesToRender = frameDependantControls == null ? 1 : numFrames;

            var totalYs = (screenshotHeight + localSize[1] - 1) / localSize[1];
            var totalXs = (screenshotWidth + localSize[0] - 1) / localSize[0];
            var stopwatch = new Stopwatch();
            for (var y = 0; y < totalYs; y++)
            {
                for (var x = 0; x < totalXs; x++)
                {
                    stopwatch.Restart();
                    for (var frame = 0; frame < framesToRender; frame++)
                    {
                        if (frameDependantControls != null)
                            frameDependantControls.Frame = frame;
                        displayInformation(string.Format("Screenshot {0}% done", 100 * (y * totalXs * framesToRender + x * framesToRender + frame) / (totalXs * totalYs * framesToRender)));

                        _kernel.Render(computeBuffer, queue, _parameters, new Size(screenshotWidth, screenshotHeight), slowRenderPower, new Size(x, y), (int)localSize[0]);
                    }

                    var pixels = new Vector4[localSize[0] * localSize[1]];
                    queue.ReadFromBuffer(computeBuffer, ref pixels, true, 0, 0, localSize[0] * localSize[1], null);
                    queue.Finish();

                    stopwatch.Stop();
                    var elapsed = stopwatch.Elapsed.TotalMilliseconds / framesToRender;
                    _kernel.AverageKernelTime = (elapsed + _kernel.AverageKernelTime * 4) / 5;

                    var blockWidth = Math.Min(localSize[0], screenshotWidth - x * localSize[0]);
                    var blockHeight = Math.Min(localSize[1], screenshotHeight - y * localSize[1]);
                    var intPixels = new byte[blockWidth * blockHeight * 3];
                    for (var py = 0; py < blockHeight; py++)
                    {
                        for (var px = 0; px < blockWidth; px++)
                        {
                            var pixel = pixels[py * localSize[1] + px];
                            if (float.IsNaN(pixel.X) || float.IsNaN(pixel.Y) || float.IsNaN(pixel.Z))
                                nancount++;
                            // BGR
                            if (float.IsNaN(pixel.Z) == false)
                                intPixels[(py * blockWidth + px) * 3 + 0] = (byte)(pixel.Z * 255);
                            if (float.IsNaN(pixel.Y) == false)
                                intPixels[(py * blockWidth + px) * 3 + 1] = (byte)(pixel.Y * 255);
                            if (float.IsNaN(pixel.X) == false)
                                intPixels[(py * blockWidth + px) * 3 + 2] = (byte)(pixel.X * 255);
                        }
                    }
                    for (var line = 0; line < blockHeight; line++)
                        Marshal.Copy(intPixels, line * (int)blockWidth * 3, new IntPtr(scan0 + ((y * localSize[1] + line) * bmpData.Stride) + x * localSize[0] * 3), (int)blockWidth * 3);
                }
            }

            bmp.UnlockBits(bmpData);
            if (nancount != 0)
                MessageBox.Show(string.Format("Caught {0} NAN pixels while taking screenshot", nancount), "Warning");

            _kernelInUse--;
            return bmp;
        }

        public void TakeGif(IGifableControl control, Action<string> displayInformation)
        {
            _kernelInUse++;
            var encoder = new AnimatedGifEncoder();
            encoder.Start(Ext.UniqueFilename("sequence", "gif"));
            encoder.SetDelay(1000 / StaticSettings.Fetch.GifFramerate);
            encoder.SetRepeat(0);
            var endIgnoreControl = control.StartIgnoreControl();

            var ccontext = _kernel.ComputeContext;
            var queue = new ComputeCommandQueue(ccontext, ccontext.Devices[0], ComputeCommandQueueFlags.None);
            var screenshotHeight = StaticSettings.Fetch.GifHeight;
            var screenshotWidth = (int)(screenshotHeight * ScreenshotAspectRatio);
            var computeBuffer = new ComputeBuffer<Vector4>(ccontext, ComputeMemoryFlags.ReadWrite, screenshotWidth * screenshotHeight);

            var fdc = control as IFrameDependantControl;
            for (var i = 0; i < StaticSettings.Fetch.GifFramecount + 1; i++)
            {
                if (fdc != null)
                    fdc.Frame = i;
                var teardown = control.SetupGif((double)i / StaticSettings.Fetch.GifFramecount);
                _kernel.Render(computeBuffer, queue, _parameters, new Size(screenshotWidth, screenshotHeight));
                queue.Finish();
                teardown();
            }
            for (var i = 1; i < StaticSettings.Fetch.GifFramecount + 1; i++)
            {
                if (fdc != null)
                    fdc.Frame = i;
                displayInformation(string.Format("{0}% done with gif", (int)(100.0 * (i - 1) / StaticSettings.Fetch.GifFramecount)));
                var teardown = control.SetupGif((double)(i - 1) / StaticSettings.Fetch.GifFramecount);
                _kernel.Render(computeBuffer, queue, _parameters, new Size(screenshotWidth, screenshotHeight));
                if (encoder.AddFrame(Download(queue, computeBuffer, screenshotWidth, screenshotHeight)) == false)
                    throw new Exception("Could not add frame to gif");
                teardown();
            }
            endIgnoreControl();
            encoder.Finish();
            computeBuffer.Dispose();
            queue.Dispose();
            displayInformation("Done with gif");
            _kernelInUse--;
        }

        private Bitmap Download(ComputeCommandQueue queue, ComputeBuffer<Vector4> buffer, int width, int height)
        {
            var pixels = new Vector4[width * height];
            queue.ReadFromBuffer(buffer, ref pixels, true, 0, 0, width * height, null);
            queue.Finish();

            var intPixels = Array.ConvertAll(pixels, pixel =>
            {
                pixel = Vector4.Clamp(pixel, new Vector4(0), new Vector4(1));
                return (byte)(pixel.X * 255) << 16 | (byte)(pixel.Y * 255) << 8 | (byte)(pixel.Z * 255);
            });
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppRgb);
            var bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            Marshal.Copy(intPixels, 0, bmpData.Scan0, intPixels.Length);
            bmp.UnlockBits(bmpData);
            return bmp;
        }

        //public void TakeGif(IGifableControl control, Action<string> displayInformation)
        //{
        //    var encoder = new AnimatedGifEncoder();
        //    encoder.Start(Ext.UniqueFilename("sequence", "gif"));
        //    encoder.SetDelay(1000 / StaticSettings.Fetch.GifFramerate);
        //    encoder.SetRepeat(0);
        //    var endIgnoreControl = control.StartIgnoreControl();
        //    for (var i = 0; i < StaticSettings.Fetch.GifFramecount; i++)
        //    {
        //        displayInformation(string.Format("{0}% done with gif", (int)(100.0 * i / StaticSettings.Fetch.GifFramecount)));
        //        var teardown = control.SetupGif((double)i / StaticSettings.Fetch.GifFramecount);
        //        var screen = Screenshot(StaticSettings.Fetch.GifHeight, 1, s => { });
        //        if (screen != null && encoder.AddFrame(screen) == false)
        //            throw new Exception("Could not add frame to gif");
        //        teardown();
        //    }
        //    endIgnoreControl();
        //    encoder.Finish();
        //    displayInformation("Done rendering sequence");
        //}

        public void Dispose()
        {
            if (_kernel != null)
                _kernel.Dispose();
        }
    }
}