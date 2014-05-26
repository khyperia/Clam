using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using System.Xml.Linq;
using Cloo;
using OpenTK.Input;

namespace Clam
{
    interface IUpdateableParameterSet
    {
        void Update(double elapsedTime, bool isFocused, HashSet<Keys> pressedKeys);
    }

    interface ISerializableParameterSet
    {
        XElement Save(string elementName);
        void Load(XElement element);
    }

    abstract class KeyboardControlBase : IParameterSet, IUpdateableParameterSet, ISerializableParameterSet
    {
        private Dictionary<Keys, Action<float>> _bindings;
        private bool _ignoreControl;

        protected void SetBindings(Dictionary<Keys, Action<float>> bindings)
        {
            _bindings = bindings;
        }

        public void Update(double elapsedTime, bool isFocused, HashSet<Keys> pressedKeys)
        {
            if (_ignoreControl)
                return;
            if (isFocused)
            {
                var keyPressed = false;
                foreach (var binding in _bindings.Where(binding => pressedKeys.Contains(binding.Key)))
                {
                    binding.Value((float)elapsedTime);
                    keyPressed = true;
                }
                if (keyPressed)
                {
                    var fdc = this as IFrameDependantControl;
                    if (fdc != null)
                        fdc.Frame = 0;
                }
            }
            OnUpdate();
        }

        public Action StartIgnoreControl()
        {
            _ignoreControl = true;
            return () => _ignoreControl = false;
        }

        protected virtual void OnUpdate()
        {
        }

        public abstract string ControlsHelp { get; }
        public abstract void ApplyToKernel(ComputeKernel kernel, bool useDouble, ref int startIndex);
        public abstract XElement Save(string controloptions);
        public abstract void Load(XElement element);
    }
}
