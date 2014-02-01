using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml.Linq;
using Cloo;
using OpenTK;
using OpenTK.Input;

namespace Clam
{
    abstract class KeyboardControlBase : IParameterSet
    {
        protected readonly RenderWindow RenderWindow;
        private Dictionary<Key, Action<float>> _bindings;

        protected KeyboardControlBase(RenderWindow renderWindow)
        {
            RenderWindow = renderWindow;
            renderWindow.Keyboard.KeyDown += KeyboardOnKeyDown;
            renderWindow.UpdateFrame += RenderWindowOnUpdateFrame;
        }

        private void RenderWindowOnUpdateFrame(object sender, FrameEventArgs frameEventArgs)
        {
            var keyPressed = false;
            foreach (var binding in _bindings.Where(binding => RenderWindow.Keyboard[binding.Key]))
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
                        o => RenderWindow.Renderer.Screenshot(StaticSettings.ScreenshotHeight, StaticSettings.ScreenshotPartialRender).Save(Ext.UniqueFilename("screenshot", "png")));
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
            OnKeyDown(keyboardKeyEventArgs.Key);
        }

        protected virtual void OnUpdate(double time)
        {
        }

        protected virtual void OnKeyDown(Key key)
        {
        }

        public abstract void ApplyToKernel(ComputeKernel kernel, ref int startIndex);
        protected abstract XElement Save();
        protected abstract void Load(XElement element);

        public void UnRegister()
        {
            RenderWindow.Keyboard.KeyDown -= KeyboardOnKeyDown;
            RenderWindow.UpdateFrame -= RenderWindowOnUpdateFrame;
        }
    }
}
