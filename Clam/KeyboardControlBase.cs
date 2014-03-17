using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Input;
using System.Xml.Linq;
using Cloo;

namespace Clam
{
    interface IUpdateableParameterSet
    {
        void Update(double elapsedTime, bool isFocused);
    }

    interface ISerializableParameterSet
    {
        XElement Save(string elementName);
        void Load(XElement element);
    }

    abstract class KeyboardControlBase : IParameterSet, IUpdateableParameterSet, ISerializableParameterSet
    {
        private Dictionary<Key, Action<float>> _bindings;
        private bool _ignoreControl;

        protected void SetBindings(Dictionary<Key, Action<float>> bindings)
        {
            _bindings = bindings;
        }

        public void Update(double elapsedTime, bool isFocused)
        {
            if (_ignoreControl)
                return;
            if (isFocused)
            {
                var keyPressed = false;
                foreach (var binding in _bindings.Where(binding => Keyboard.IsKeyDown(binding.Key)))
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
