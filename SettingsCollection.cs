using Eto.Forms;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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

        internal SettingsCollection Clone()
        {
            var result = new SettingsCollection();
            foreach (var kvp in this)
            {
                result[kvp.Key] = kvp.Value;
            }
            return result;
        }

        public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => _settings.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        internal void Save(string fileName)
        {
            File.WriteAllLines(fileName, this.Select(kvp => $"{kvp.Key} = {kvp.Value}"));
        }

        internal void Load(string fileName)
        {
            try
            {
                foreach (var line in File.ReadLines(fileName))
                {
                    var index = line.IndexOf('=');
                    if (index == -1)
                    {
                        throw new Exception("Invalid file format: no '=' character in line");
                    }
                    var key = line.Substring(0, index).Trim();
                    var value = line.Substring(index + 1).Trim();
                    var existing = this[key];
                    if (existing is bool)
                    {
                        this[key] = bool.Parse(value);
                    }
                    else if (existing is int)
                    {
                        this[key] = int.Parse(value);
                    }
                    else if (existing is float)
                    {
                        this[key] = float.Parse(value);
                    }
                    else if (existing is double)
                    {
                        this[key] = double.Parse(value);
                    }
                    else
                    {
                        throw new Exception("Invalid type in config: " + existing.GetType().Name);
                    }
                }
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message, e.GetType().Name);
            }
        }
    }
}