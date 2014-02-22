using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml.Linq;
using Clam.NGif;
using Cloo;
using OpenTK;
using OpenTK.Input;

namespace Clam
{
    interface IGifableControl
    {
        // pointInFrame = range(0, 1)
        // return value = teardown action
        Action SetupGif(double pointInFrame);
    }

    abstract class KeyboardControlBase : IParameterSet
    {
        private readonly RenderWindow _renderWindow;
        private Dictionary<Key, Action<float>> _bindings;

        protected KeyboardControlBase(RenderWindow renderWindow)
        {
            _renderWindow = renderWindow;
            renderWindow.Keyboard.KeyDown += KeyboardOnKeyDown;
            renderWindow.UpdateFrame += RenderWindowOnUpdateFrame;
        }

        private void RenderWindowOnUpdateFrame(object sender, FrameEventArgs frameEventArgs)
        {
            var keyPressed = false;
            foreach (var binding in _bindings.Where(binding => _renderWindow.Keyboard[binding.Key]))
            {
                binding.Value((float)frameEventArgs.Time);
                keyPressed = true;
            }
            if (keyPressed)
            {
                var fdc = this as IFrameDependantControl;
                if (fdc != null)
                    fdc.Frame = 0;
            }
            OnUpdate(frameEventArgs.Time);
        }

        protected void SetBindings(Dictionary<Key, Action<float>> bindings)
        {
            _bindings = bindings;
        }

        private void KeyboardOnKeyDown(object sender, KeyboardKeyEventArgs keyboardKeyEventArgs)
        {
            switch (keyboardKeyEventArgs.Key)
            {
                case Key.P:
                    ThreadPool.QueueUserWorkItem(
                        o => _renderWindow.Renderer.Screenshot(StaticSettings.ScreenshotHeight, StaticSettings.ScreenshotPartialRender).Save(Ext.UniqueFilename("screenshot", "png")));
                    break;
                case Key.O:
                    var gifable = this as IGifableControl;
                    if (gifable != null)
                        ThreadPool.QueueUserWorkItem(o => TakeGif(_renderWindow.Renderer, gifable));
                    break;
                case Key.L:
                    var filename = Path.Combine("..", "State", GetType().Name + ".state.xml");
                    var directory = Path.GetDirectoryName(filename);
                    if (directory != null && Directory.Exists(directory) == false)
                        Directory.CreateDirectory(directory);
                    Save().Save(filename);
                    RenderWindow.DisplayInformation("Saved state");
                    break;
                case Key.K:
                    try
                    {
                        Load(XDocument.Load(Path.Combine("..", "State", GetType().Name + ".state.xml")).Root);
                        var fdc = this as IFrameDependantControl;
                        if (fdc != null)
                            fdc.Frame = 0;
                        RenderWindow.DisplayInformation("Loaded state");
                    }
                    catch (Exception e)
                    {
                        RenderWindow.DisplayInformation("Could not load state: " + e.GetType().Name + ": " + e.Message);
                    }
                    break;
            }
        }

        private static void TakeGif(RenderPackage renderer, IGifableControl control)
        {
            // TODO: Add to StaticSettings
            var encoder = new AnimatedGifEncoder();
            encoder.Start(Ext.UniqueFilename("sequence", "gif"));
            encoder.SetDelay(1000 / StaticSettings.GifFramerate);
            encoder.SetRepeat(0);
            for (var i = 0; i < StaticSettings.GifFramecount; i++)
            {
                var teardown = control.SetupGif((double)i / StaticSettings.GifFramecount);
                var screen = renderer.Screenshot(StaticSettings.GifHeight, 1);
                if (encoder.AddFrame(screen) == false)
                    throw new Exception("Could not add frame to gif");
                teardown();
            }
            encoder.Finish();
            RenderWindow.DisplayInformation("Done rendering sequence");
        }

        protected virtual void OnUpdate(double time)
        {
        }

        public abstract void ApplyToKernel(ComputeKernel kernel, ref int startIndex);
        protected abstract XElement Save();
        protected abstract void Load(XElement element);

        public void UnRegister()
        {
            _renderWindow.Keyboard.KeyDown -= KeyboardOnKeyDown;
            _renderWindow.UpdateFrame -= RenderWindowOnUpdateFrame;
        }
    }
}
