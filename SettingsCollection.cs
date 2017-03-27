using System;
using System.Collections;
using System.Collections.Generic;

namespace Clam4
{
    public class SettingsCollection : IEnumerable<KeyValuePair<string, object>>
    {
        private readonly Dictionary<string, object> _settings = new Dictionary<string, object>();
        public event Action<string, object> OnChange;

        public SettingsCollection()
        {
        }

        public bool TryGet(string name, out object value)
        {
            return _settings.TryGetValue(name, out value);
        }

        public object this[string name]
        {
            get { return _settings[name]; }
            set
            {
                if (!_settings.TryGetValue(name, out var existing) || !existing.Equals(value))
                {
                    _settings[name] = value;
                    OnChange?.Invoke(name, value);
                }
            }
        }

        public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => _settings.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}