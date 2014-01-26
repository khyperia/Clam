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
                {Key.Left, dt => _x -= _zoom / 100},
                {Key.Right, dt => _x += _zoom / 100},
                {Key.Up, dt => _y -= _zoom / 100},
                {Key.Down, dt => _y += _zoom / 100},
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
                new XElement("X", _x),
                new XElement("Y", _y),
                new XElement("Zoom", _zoom));
        }

        protected override void Load(XElement element)
        {
            _x = double.Parse(element.Element("X").Value);
            _y = double.Parse(element.Element("Y").Value);
            _zoom = double.Parse(element.Element("Zoom").Value);
        }
    }
}
