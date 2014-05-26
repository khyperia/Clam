using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.Xml.Linq;
using Cloo;

namespace Clam
{
    class Keyboard2DControl : KeyboardControlBase, IGifableControl
    {
        private double _x;
        private double _y;
        private double _zoom = 1;

        public Keyboard2DControl()
        {
            SetBindings(new Dictionary<Keys, Action<float>>
            {
                {Keys.J, dt => _x -= _zoom / 2 * dt},
                {Keys.L, dt => _x += _zoom / 2 * dt},
                {Keys.I, dt => _y -= _zoom / 2 * dt},
                {Keys.K, dt => _y += _zoom / 2 * dt},
                {Keys.W, dt => _zoom /= dt + 1},
                {Keys.S, dt => _zoom *= dt + 1},
            });
        }

        public override string ControlsHelp
        {
            get { return "Navigation: I/J/K/L\nZooming: W/S"; }
        }

        public override void ApplyToKernel(ComputeKernel kernel, bool isDouble, ref int startIndex)
        {
            if (isDouble)
            {
                kernel.SetValueArgument(startIndex++, _x);
                kernel.SetValueArgument(startIndex++, _y);
                kernel.SetValueArgument(startIndex++, _zoom);
            }
            else
            {
                kernel.SetValueArgument(startIndex++, (float)_x);
                kernel.SetValueArgument(startIndex++, (float)_y);
                kernel.SetValueArgument(startIndex++, (float)_zoom);
            }
        }

        public override XElement Save(string elementName)
        {
            return new XElement(elementName,
                _x.Save("X"),
                _y.Save("Y"),
                _zoom.Save("Zoom"));
        }

        public override void Load(XElement element)
        {
            _x = element.Element("X").LoadDouble();
            _y = element.Element("Y").LoadDouble();
            _zoom = element.Element("Zoom").LoadDouble();
        }

        public Action SetupGif(double pointInFrame)
        {
            var oldZoom = _zoom;
            _zoom *= Math.Pow(0.0001, pointInFrame);
            return () => _zoom = oldZoom;
        }
    }
}
