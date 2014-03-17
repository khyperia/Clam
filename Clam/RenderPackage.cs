using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Xml.Linq;
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

        public RenderPackage([NotNull] RenderKernel kernel, [NotNull] IParameterSet parameters)
        {
            _kernel = kernel;
            _parameters = parameters;
        }

        public RenderKernel Kernel
        {
            get { return _kernel; }
        }

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
        };

        public static RenderPackage? LoadFromXml(ComputeContext computeContext, string xmlFilename)
        {
            var xml = XDocument.Load(xmlFilename).Root;
            if (xml == null)
                throw new Exception("Invalid XML file " + xmlFilename + ": no root element");
            var controls = xml.Element("Controls");
            if (controls == null)
                throw new Exception("Invalid XML file " + xmlFilename + ": no Controls element");
            if (!ControlBindings.ContainsKey(controls.Value))
                throw new Exception("Invalid XML file " + xmlFilename + ": Controls value did not exist, possible values are: " + string.Join(", ", ControlBindings.Keys));
            var parameters = ControlBindings[controls.Value]();
            var files = xml.Element("Files");
            if (files == null)
                throw new Exception("Invalid XML file " + xmlFilename + ": no Files element");
            var sourceStrings = files.Elements("File").Select(e => File.ReadAllText(e.Value)).ToArray();
            var kernel = RenderKernel.Create(computeContext, sourceStrings);
            if (kernel == null)
                return null;
            return new RenderPackage(kernel, parameters);
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
                bmp = new Bitmap(screenshotWidth, screenshotHeight);
            }
            catch (ArgumentException)
            {
                MessageBox.Show("Image size too big", "Error");
                return null;
            }
            var nancount = 0;
            var bmpData = bmp.LockBits(new Rectangle(0, 0, screenshotWidth, screenshotHeight), ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            var scan0 = bmpData.Scan0.ToInt64();
            var queue = new ComputeCommandQueue(ccontext, ccontext.Devices[0], ComputeCommandQueueFlags.None);
            var localSize = _kernel.Threadsize(queue);
            for (var i = 0; i < localSize.Length; i++)
                localSize[i] *= slowRenderPower;
            var computeBuffer = new ComputeBuffer<Vector4>(ccontext, ComputeMemoryFlags.ReadWrite, localSize[0] * localSize[1]);

            const int numFrames = 150;
            var frameDependantControls = _parameters as IFrameDependantControl;
            var framesToRender = frameDependantControls == null ? 1 : numFrames;

            var totalYs = (screenshotHeight + localSize[1] - 1) / localSize[1];
            var totalXs = (screenshotWidth + localSize[0] - 1) / localSize[0];
            for (var y = 0; y < totalYs; y++)
            {
                for (var x = 0; x < totalXs; x++)
                {
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

                    var blockWidth = Math.Min(localSize[0], screenshotWidth - x * localSize[0]);
                    var blockHeight = Math.Min(localSize[1], screenshotHeight - y * localSize[1]);
                    var intPixels = new int[blockWidth * blockHeight];
                    for (var py = 0; py < blockHeight; py++)
                    {
                        for (var px = 0; px < blockWidth; px++)
                        {
                            var pixel = pixels[py * localSize[1] + px];
                            int result;
                            if (float.IsNaN(pixel.X) || float.IsNaN(pixel.Y) || float.IsNaN(pixel.Z))
                            {
                                nancount++;
                                result = 0;
                            }
                            else
                                result = (byte)(pixel.X * 255) << 16 | (byte)(pixel.Y * 255) << 8 | (byte)(pixel.Z * 255);
                            intPixels[py * blockWidth + px] = result;
                        }
                    }
                    for (var line = 0; line < blockHeight; line++)
                        Marshal.Copy(intPixels, line * (int)blockWidth,
                            new IntPtr(scan0 + ((y * localSize[1] + line) * screenshotWidth + x * localSize[0]) * sizeof(int)), (int)blockWidth);
                }
            }

            bmp.UnlockBits(bmpData);
            if (nancount != 0)
                MessageBox.Show(string.Format("Caught {0} NAN pixels while taking screenshot", nancount), "Warning");

            displayInformation("Done rendering screenshot");
            _kernelInUse--;
            return bmp;
        }

        public void TakeGif(IGifableControl control, Action<string> displayInformation)
        {
            var encoder = new AnimatedGifEncoder();
            encoder.Start(Ext.UniqueFilename("sequence", "gif"));
            encoder.SetDelay(1000 / StaticSettings.Fetch.GifFramerate);
            encoder.SetRepeat(0);
            var endIgnoreControl = control.StartIgnoreControl();
            for (var i = 0; i < StaticSettings.Fetch.GifFramecount; i++)
            {
                displayInformation(string.Format("{0}% done with gif", (int)(100.0 * i / StaticSettings.Fetch.GifFramecount)));
                var teardown = control.SetupGif((double)i / StaticSettings.Fetch.GifFramecount);
                var screen = Screenshot(StaticSettings.Fetch.GifHeight, 1, s => { });
                if (screen != null && encoder.AddFrame(screen) == false)
                    throw new Exception("Could not add frame to gif");
                teardown();
            }
            endIgnoreControl();
            encoder.Finish();
            displayInformation("Done rendering sequence");
        }

        public void Dispose()
        {
            if (_kernel != null)
                _kernel.Dispose();
        }
    }
}