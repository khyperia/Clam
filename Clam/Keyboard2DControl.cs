using System;
using System.Collections.Generic;
using System.Windows.Input;
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

        public override string ControlsHelp
        {
            get { return "Navigation: Up/Down/Left/Right\nZooming: W/S"; }
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
