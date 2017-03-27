using System;
using System.IO;
using System.Reflection;
using ManagedCuda.NVRTC;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Clam4
{
    internal sealed class CodeObject
    {
        private readonly string _resourceName;
        private byte[] _cache;

        public CodeObject(string resourceName)
        {
            _resourceName = resourceName;
        }

        private static Stream LoadResource(string resourceName)
        {
            var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(typeof(Kernel), resourceName);
            return stream ?? throw new Exception("Resource not found: " + resourceName);
        }

        private static string LoadResourceString(string resourceName)
        {
            using (var reader = new StreamReader(LoadResource(resourceName)))
            {
                return reader.ReadToEnd();
            }
        }

        private static void Headers(out string[] names, out string[] sources)
        {
            var resourceNames = new List<string>();
            foreach (var file in Assembly.GetExecutingAssembly().GetManifestResourceNames())
            {
                if (file.EndsWith(".h"))
                {
                    resourceNames.Add(file);
                }
            }
            names = new string[resourceNames.Count];
            sources = new string[resourceNames.Count];
            for (var i = 0; i < resourceNames.Count; i++)
            {
                names[i] = resourceNames[i].Substring(resourceNames[i].IndexOf('.') + 1);
                sources[i] = LoadResourceString(names[i]);
            }
        }

        private static byte[] Compile(string resourceName)
        {
            string source = LoadResourceString(resourceName);
            Headers(out var names, out var sources);
            using (var compiler = new CudaRuntimeCompiler(source, resourceName, names, sources))
            {
                var error = false;
                try
                {
                    compiler.Compile(new[] {
                        "--use_fast_math",
                        // "--generate-line-info",
                        // "--gpu-architecture=sm_52",
                        // "--resource-usage",
                        // "--Werror",
                    });
                }
                catch (NVRTCException)
                {
                    error = true;
                }
                var log = compiler.GetLogAsString();
                if (error || !string.IsNullOrEmpty(log))
                {
                    MessageBox.Show(log, error ? "Compiler error" : "Compiler warning");
                }
                if (error)
                {
                    return null;
                }
                return compiler.GetPTX();
            }
        }

        public byte[] PTX
        {
            get
            {
                if (_cache == null)
                {
                    var ext = Path.GetExtension(_resourceName);
                    switch (ext)
                    {
                        case ".ptx":
                            using (var memstream = new MemoryStream())
                            using (var resource = LoadResource(_resourceName))
                            {
                                resource.CopyTo(memstream);
                                _cache = memstream.ToArray();
                            }
                            break;
                        case ".cu":
                            _cache = Compile(_resourceName);
                            break;
                        default:
                            throw new Exception($"Unknown filetype: {_resourceName}");
                    }
                }
                return _cache;
            }
        }
    }
}