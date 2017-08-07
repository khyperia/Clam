using Eto.Forms;
using System.Collections.Generic;
using System.Diagnostics;

namespace Clam4
{
    internal class KeyTracker
    {
        private readonly object lock_object = new object();
        private readonly HashSet<Keys> pressedKeys = new HashSet<Keys>();
        private readonly IKeyHandler[] _keyHandlers;
        private readonly SettingsCollection _settings;
        private Stopwatch _stopwatch;

        public KeyTracker(IKeyHandler[] keyHandlers, SettingsCollection mutate)
        {
            _keyHandlers = keyHandlers;
            _settings = mutate;
            _stopwatch = Stopwatch.StartNew();
        }

        public SettingsCollection Settings => _settings;

        public bool OnKeyDown(Keys keys)
        {
            lock (lock_object)
            {
                Invoke(Keys.None);
                pressedKeys.Add(keys);
                return InvokeWithDt(keys, 0.0);
            }
        }

        public bool OnKeyUp(Keys keys)
        {
            lock (lock_object)
            {
                var handled = Invoke(keys);
                pressedKeys.Remove(keys);
                return handled;
            }
        }

        public void OnRender()
        {
            lock (lock_object)
            {
                Invoke(Keys.None);
            }
        }

        private bool Invoke(Keys checking)
        {
            var dt = _stopwatch.Elapsed.TotalSeconds;
            _stopwatch.Restart();
            return InvokeWithDt(checking, dt);
        }

        private bool InvokeWithDt(Keys checking, double dt)
        {
            var checkGood = false;
            foreach (var key in pressedKeys)
            {
                foreach (var keyHandler in _keyHandlers)
                {
                    var handled = keyHandler.Handle(key, dt, _settings);
                    if (key == checking && handled)
                    {
                        checkGood = true;
                    }
                }
            }
            return checkGood;
        }
    }
}