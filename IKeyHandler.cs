using System;
using System.Windows.Forms;

namespace Clam4
{
    internal interface IKeyHandler
    {
        bool Handle(Keys key, double dt, SettingsCollection settings);
    }

    internal class KeyPan2d : IKeyHandler
    {
        private readonly Keys _up;
        private readonly Keys _down;
        private readonly Keys _left;
        private readonly Keys _right;
        private readonly Keys _zoomin;
        private readonly Keys _zoomout;
        private readonly string _posX;
        private readonly string _posY;
        private readonly string _zoom;

        public KeyPan2d(Keys up, Keys down, Keys left, Keys right, Keys zoomin, Keys zoomout, string posX, string posY, string zoom)
        {
            _up = up;
            _down = down;
            _left = left;
            _right = right;
            _zoomin = zoomin;
            _zoomout = zoomout;
            _posX = posX;
            _posY = posY;
            _zoom = zoom;
        }

        public bool Handle(Keys key, double dt, SettingsCollection settings)
        {
            var panSpeed = (double)settings[_zoom] * dt * 0.5f;
            var zoomSpeed = dt * 1.0f;
            if (key == _up)
            {
                settings[_posY] = (double)settings[_posY] - panSpeed;
            }
            else if (key == _down)
            {
                settings[_posY] = (double)settings[_posY] + panSpeed;
            }
            else if (key == _left)
            {
                settings[_posX] = (double)settings[_posX] - panSpeed;
            }
            else if (key == _right)
            {
                settings[_posX] = (double)settings[_posX] + panSpeed;
            }
            else if (key == _zoomin)
            {
                settings[_zoom] = (double)settings[_zoom] / (1 + zoomSpeed);
            }
            else if (key == _zoomout)
            {
                settings[_zoom] = (double)settings[_zoom] * (1 + zoomSpeed);
            }
            else
            {
                return false;
            }
            return true;
        }
    }

    internal class KeyCamera3d : IKeyHandler
    {
        private readonly string _posX;
        private readonly string _posY;
        private readonly string _posZ;
        private readonly string _lookX;
        private readonly string _lookY;
        private readonly string _lookZ;
        private readonly string _upX;
        private readonly string _upY;
        private readonly string _upZ;
        private readonly string _moveSpeed;
        private readonly string _fov;

        public KeyCamera3d(
                string posX,
                string posY,
                string posZ,
                string lookX,
                string lookY,
                string lookZ,
                string upX,
                string upY,
                string upZ,
                string moveSpeed,
                string fov
            )
        {
            _posX = posX;
            _posY = posY;
            _posZ = posZ;
            _lookX = lookX;
            _lookY = lookY;
            _lookZ = lookZ;
            _upX = upX;
            _upY = upY;
            _upZ = upZ;
            _moveSpeed = moveSpeed;
            _fov = fov;
        }

        public bool Handle(Keys key, double dt, SettingsCollection settings)
        {
            Float3 pos = new Float3((float)settings[_posX], (float)settings[_posY], (float)settings[_posZ]);
            Float3 look = new Float3((float)settings[_lookX], (float)settings[_lookY], (float)settings[_lookZ]);
            Float3 up = new Float3((float)settings[_upX], (float)settings[_upY], (float)settings[_upZ]);
            float moveSpeed = (float)settings[_moveSpeed] * (float)dt;
            float turnSpeed = (float)settings[_fov] * (float)dt;
            float rollSpeed = (float)dt;
            const Keys forwards = Keys.W;
            const Keys back = Keys.S;
            const Keys upKey = Keys.Space;
            const Keys down = Keys.Z;
            const Keys right = Keys.D;
            const Keys left = Keys.A;
            const Keys pitchUp = Keys.I;
            const Keys pitchUpTwo = Keys.Up;
            const Keys pitchDown = Keys.K;
            const Keys pitchDownTwo = Keys.Down;
            const Keys yawLeft = Keys.J;
            const Keys yawLeftTwo = Keys.Left;
            const Keys yawRight = Keys.L;
            const Keys yawRightTwo = Keys.Right;
            const Keys rollLeftOne = Keys.U;
            const Keys rollLeftTwo = Keys.Q;
            const Keys rollRightOne = Keys.O;
            const Keys rollRightTwo = Keys.E;
            if (key == forwards)
            {
                pos = pos + look * moveSpeed;
            }
            else if (key == back)
            {
                pos = pos - look * moveSpeed;
            }
            else if (key == upKey)
            {
                pos = pos + up * moveSpeed;
            }
            else if (key == down)
            {
                pos = pos - up * moveSpeed;
            }
            else if (key == right)
            {
                pos = pos + Float3.Cross(look, up) * moveSpeed;
            }
            else if (key == left)
            {
                pos = pos - Float3.Cross(look, up) * moveSpeed;
            }
            else if (key == pitchUp || key == pitchUpTwo)
            {
                look = look + up * turnSpeed;
            }
            else if (key == pitchDown || key == pitchDownTwo)
            {
                look = look - up * turnSpeed;
            }
            else if (key == yawRight || key == yawRightTwo)
            {
                look = look + Float3.Cross(look, up) * turnSpeed;
            }
            else if (key == yawLeft || key == yawLeftTwo)
            {
                look = look - Float3.Cross(look, up) * turnSpeed;
            }
            else if (key == rollRightOne || key == rollRightTwo)
            {
                up = up + Float3.Cross(look, up) * rollSpeed;
            }
            else if (key == rollLeftOne || key == rollLeftTwo)
            {
                up = up - Float3.Cross(look, up) * rollSpeed;
            }
            else
            {
                return false;
            }
            look = look.Normalized;
            up = Float3.Cross(Float3.Cross(look, up), look).Normalized;
            settings[_posX] = pos.X;
            settings[_posY] = pos.Y;
            settings[_posZ] = pos.Z;
            settings[_lookX] = look.X;
            settings[_lookY] = look.Y;
            settings[_lookZ] = look.Z;
            settings[_upX] = up.X;
            settings[_upY] = up.Y;
            settings[_upZ] = up.Z;
            return true;
        }

        public void NormalizeLookUp(SettingsCollection settings)
        {
            Float3 pos = new Float3((float)settings[_posX], (float)settings[_posY], (float)settings[_posZ]);
            Float3 look = new Float3((float)settings[_lookX], (float)settings[_lookY], (float)settings[_lookZ]);
            Float3 up = new Float3((float)settings[_upX], (float)settings[_upY], (float)settings[_upZ]);
            look = look.Normalized;
            up = Float3.Cross(Float3.Cross(look, up), look).Normalized;
            settings[_posX] = pos.X;
            settings[_posY] = pos.Y;
            settings[_posZ] = pos.Z;
            settings[_lookX] = look.X;
            settings[_lookY] = look.Y;
            settings[_lookZ] = look.Z;
            settings[_upX] = up.X;
            settings[_upY] = up.Y;
            settings[_upZ] = up.Z;
        }
    }

    internal class KeyScale : IKeyHandler
    {
        Keys _increase;
        Keys _decrease;
        string _settingName;
        double _scaleFactor;

        public KeyScale(Keys increase, Keys decrease, string settingName, double scaleFactor)
        {
            _increase = increase;
            _decrease = decrease;
            _settingName = settingName;
            _scaleFactor = scaleFactor;
        }

        public bool Handle(Keys key, double dt, SettingsCollection settings)
        {
            var value = (double)(float)settings[_settingName];
            var handled = false;
            if (_scaleFactor > 0)
            {
                if (key == _increase)
                {
                    value += _scaleFactor;
                    handled = true;
                }
                else if (key == _decrease)
                {
                    value -= dt * _scaleFactor;
                    handled = true;
                }
            }
            else if (_scaleFactor < 0)
            {
                if (key == _increase)
                {
                    value *= Math.Exp(dt * -_scaleFactor);
                    handled = true;
                }
                else if (key == _decrease)
                {
                    value *= Math.Exp(dt * _scaleFactor);
                    handled = true;
                }
            }
            if (handled)
            {
                settings[_settingName] = (float)value;
                return true;
            }
            return false;
        }
    }
}
