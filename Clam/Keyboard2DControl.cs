using System;
using System.Collections.Generic;
using System.Xml.Linq;
using Cloo;
using OpenTK.Input;

namespace Clam
{
    class Keyboard2DControl : KeyboardControlBase
    {
        private double _x;
        private double _y;
        private double _zoom = 1;

        public Keyboard2DControl(RenderWindow renderWindow)
            : base(renderWindow)
        {
            SetBindings(new Dictionary<Key, Action<float>>
            {
                {Key.Left, dt => _x -= _zoom / 2 * dt},
                {Key.Right, dt => _x += _zoom / 2 * dt},
                {Key.Up, dt => _y -= _zoom / 2 * dt},
                {Key.Down, dt => _y += _zoom / 2 * dt},
                {Key.W, dt => _zoom /= dt + 1},
                {Key.S, dt => _zoom *= dt + 1},
            });
        }

        public override void ApplyToKernel(ComputeKernel kernel, ref int startIndex)
        {
            kernel.SetValueArgument(startIndex++, (float)_x);
            kernel.SetValueArgument(startIndex++, (float)_y);
            kernel.SetValueArgument(startIndex++, (float)_zoom);
        }

        protected override XElement Save()
        {
            return new XElement("Keyboard2DControl",
                _x.Save("X"),
                _y.Save("Y"),
                _zoom.Save("Zoom"));
        }

        protected override void Load(XElement element)
        {
            _x = element.Element("X").LoadDouble();
            _y = element.Element("Y").LoadDouble();
            _zoom = element.Element("Zoom").LoadDouble();
        }
    }
}
