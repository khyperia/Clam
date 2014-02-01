using System;
using System.Collections.Generic;
using System.Xml.Linq;
using Cloo;
using OpenTK;
using OpenTK.Input;

namespace Clam
{
    interface IFrameDependantControl
    {
        int Frame { get; set; }
    }

    class KeyboardRaytracerControlExtended : KeyboardRaytracerControl, IFrameDependantControl
    {
        public int Frame { get; set; }

        public KeyboardRaytracerControlExtended(RenderWindow renderWindow)
            : base(renderWindow)
        {
        }

        public override void ApplyToKernel(ComputeKernel kernel, ref int param)
        {
            base.ApplyToKernel(kernel, ref param);
            kernel.SetValueArgument(param++, Fov);
            kernel.SetValueArgument(param++, MoveSpeed * 3);
            kernel.SetValueArgument(param++, Frame);
        }
    }

    class KeyboardRaytracerControl : KeyboardControlBase, IGifableControl
    {
        private const float TurnSpeed = 1f;
        private Vector3d _position = new Vector3d(4, 0, 0);
        private Vector3d _lookat = new Vector3d(-1, 0, 0);
        private Vector3d _up = new Vector3d(0, 1, 0);
        protected float MoveSpeed = 1;
        protected float Fov = 1;

        public KeyboardRaytracerControl(RenderWindow renderWindow)
            : base(renderWindow)
        {
            SetBindings(new Dictionary<Key, Action<float>>
            {
                {Key.W, dt => _position += _lookat * dt * MoveSpeed},
                {Key.S, dt => _position -= _lookat * dt * MoveSpeed},
                {Key.A, dt => _position += Vector3d.Cross(_up, _lookat) * dt * MoveSpeed},
                {Key.D, dt => _position -= Vector3d.Cross(_up, _lookat) * dt * MoveSpeed},
                {Key.ShiftLeft, dt => _position += _up * dt * MoveSpeed},
                {Key.Space, dt => _position -= _up * dt * MoveSpeed},
                {Key.Q, dt => _up = Vector3d.Transform(_up, Matrix4d.CreateFromAxisAngle(_lookat, TurnSpeed * dt))},
                {Key.E, dt => _up = Vector3d.Transform(_up, Matrix4d.CreateFromAxisAngle(_lookat, -TurnSpeed * dt))},
                {Key.Left, dt => _lookat = Vector3d.Transform(_lookat, Matrix4d.CreateFromAxisAngle(_up, TurnSpeed * dt * Fov))},
                {Key.Right, dt => _lookat = Vector3d.Transform(_lookat, Matrix4d.CreateFromAxisAngle(_up, -TurnSpeed * dt * Fov))},
                {Key.Up, dt => _lookat = Vector3d.Transform(_lookat, Matrix4d.CreateFromAxisAngle(Vector3d.Cross(_up, _lookat), TurnSpeed * dt * Fov))},
                {Key.Down, dt => _lookat = Vector3d.Transform(_lookat, Matrix4d.CreateFromAxisAngle(Vector3d.Cross(_up, _lookat), -TurnSpeed * dt * Fov))},
                {Key.R, dt => MoveSpeed *= 1 + dt},
                {Key.F, dt =>  MoveSpeed *= 1 - dt},
                {Key.N, dt => Fov *= 1 + dt},
                {Key.M, dt => Fov *= 1 - dt}
            });
        }

        protected override void OnUpdate(double time)
        {
            _up = Vector3d.Cross(Vector3d.Cross(_lookat, _up), _lookat);
            _lookat = Vector3d.Normalize(_lookat);
            _up = Vector3d.Normalize(_up);
        }

        public override void ApplyToKernel(ComputeKernel kernel, ref int startIndex)
        {
            kernel.SetValueArgument(startIndex++, new Vector4((Vector3)_position));
            kernel.SetValueArgument(startIndex++, new Vector4((Vector3)_lookat));
            kernel.SetValueArgument(startIndex++, new Vector4((Vector3)_up));
        }

        protected override XElement Save()
        {
            return new XElement("KeyboardRaytracerControl",
                _position.Save("Position"),
                _lookat.Save("Lookat"),
                _up.Save("Up"),
                MoveSpeed.Save("MoveSpeed"),
                Fov.Save("Fov")
                );
        }

        protected override void Load(XElement element)
        {
            _position = element.Element("Position").LoadVector3D();
            _lookat = element.Element("Lookat").LoadVector3D();
            _up = element.Element("Up").LoadVector3D();
            MoveSpeed = element.Element("MoveSpeed").LoadFloat();
            Fov = element.Element("Fov").LoadFloat();
        }

        public Action SetupGif(double pointInFrame)
        {
            var oldPosition = _position;
            _position += Vector3d.Cross(_lookat, _up) * Math.Sin(pointInFrame * (Math.PI * 2)) * MoveSpeed / 10;
            return () => _position = oldPosition;
        }
    }
}
