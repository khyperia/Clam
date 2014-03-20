using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Xml.Linq;
using Cloo;
using JetBrains.Annotations;
using OpenTK;
using Size = System.Drawing.Size;

namespace Clam
{
    public interface IParameterSet
    {
        string ControlsHelp { get; }
        void ApplyToKernel(ComputeKernel kernel, bool useDouble, ref int startIndex);
        Action StartIgnoreControl();
    }

    public interface IGifableControl
    {
        // pointInFrame = range(0, 1)
        // return value = teardown action
        Action SetupGif(double pointInFrame);
        // return value = EndIgnoreControl
        Action StartIgnoreControl();
    }

    public class RenderKernel : IDisposable
    {
        private readonly ComputeContext _context;
        private readonly string[] _sourcecodes;
        private readonly Dictionary<string, string> _defines;
        private ComputeKernel _kernel;
        private readonly object _kernelLock = new object();
        private long[] _localSize;
        private bool _useDouble;

        public double AverageKernelTime { get; set; }

        private RenderKernel(ComputeContext context, ComputeKernel kernel, string[] sourcecodes, Dictionary<string, string> defines)
        {
            _context = context;
            _kernel = kernel;
            _sourcecodes = sourcecodes;
            _defines = defines;
        }

        public ComputeContext ComputeContext { get { return _context; } }

        private static ComputeKernel Compile(ComputeContext context, string[] sourcecodes, Dictionary<string, string> defines)
        {
            var program = new ComputeProgram(context, sourcecodes);
            var device = context.Devices.Single();
            try
            {
                foreach (var define in defines.Where(define => define.Key.Any(char.IsWhiteSpace) || define.Value.Any(char.IsWhiteSpace)))
                {
                    MessageBox.Show("Invalid define \"" + define.Key + "=" + define.Value + "\": define contained whitespace", "Error");
                    return null;
                }
                var options = string.Join(" ", defines.Where(kvp => !string.IsNullOrEmpty(kvp.Value)).Select(kvp => "-D " + kvp.Key + "=" + kvp.Value));
                program.Build(new[] { device }, options + " " + StaticSettings.Fetch.OpenClOptions, null, IntPtr.Zero);
                var str = program.GetBuildLog(device).Trim();
                if (string.IsNullOrEmpty(str) == false)
                    MessageBox.Show(str, "Build log");
                return program.CreateKernel("Main");
            }
            catch (InvalidBinaryComputeException)
            {
                MessageBox.Show(program.GetBuildLog(device), "Build error (invalid binary)");
                return null;
            }
            catch (BuildProgramFailureComputeException)
            {
                MessageBox.Show(program.GetBuildLog(device), "Build error (build program failure)");
                return null;
            }
        }

        private static readonly Regex DefineRegex = new Regex(@"^#ifndef +(\w+)\r?$", RegexOptions.Multiline);
        private static readonly Regex DefineDefaultRegex = new Regex(@"^#define +(\w+) +([^\r\n]+)\r?$", RegexOptions.Multiline);
        private static IEnumerable<string> CollectDefines(IEnumerable<string> sourcecodes)
        {
            return sourcecodes.SelectMany(sourcecode => DefineRegex.Matches(sourcecode).Cast<Match>(), (sourcecode, match) => match.Groups[1].Captures[0].Value);
        }

        private static IEnumerable<KeyValuePair<string, string>> CollectDefaultDefines(IEnumerable<string> sourcecodes)
        {
            return sourcecodes.SelectMany(sourcecode => DefineDefaultRegex.Matches(sourcecode).Cast<Match>(),
                (sourcecode, match) => new KeyValuePair<string, string>(match.Groups[1].Captures[0].Value, match.Groups[2].Captures[0].Value));
        }

        [CanBeNull]
        public static RenderKernel Create(ComputeContext context, string[] sourcecodes)
        {
            var defines = CollectDefines(sourcecodes).ToDictionary(define => define, define => "");
            foreach (var defaultDefine in CollectDefaultDefines(sourcecodes).Where(defaultDefine => defines.ContainsKey(defaultDefine.Key)))
                defines[defaultDefine.Key] = defaultDefine.Value;
            var compilation = Compile(context, sourcecodes, defines);
            return compilation == null ? null : new RenderKernel(context, compilation, sourcecodes, defines);
        }

        public IEnumerable<KeyValuePair<string, string>> Options
        {
            get { return _defines; }
        }

        public XElement SerializeOptions(string elementName)
        {
            return new XElement(elementName, _defines.Select(kvp => (object)new XElement(kvp.Key, kvp.Value)).ToArray());
        }

        public void LoadOptions(XElement element)
        {
            foreach (var xElement in element.Elements())
            {
                var key = xElement.Name.LocalName;
                var value = xElement.Value;
                if (_defines.ContainsKey(key))
                    _defines[key] = value;
                else
                    MessageBox.Show(string.Format("Option {0} was invalid (did you load the wrong XML?)", key), "Error");
            }
            Recompile();
        }

        public void SetOption(string key, string value)
        {
            if (_defines.ContainsKey(key) == false)
                throw new Exception("Define " + key + " does not exist, while trying to set it's value in RenderKernel");
            _defines[key] = value;
        }

        public void Recompile()
        {
            lock (_kernelLock)
            {
                var newKernel = Compile(_context, _sourcecodes, _defines);
                Dispose();
                int useDoubleDefine;
                _useDouble = _defines.ContainsKey("UseDouble") && int.TryParse(_defines["UseDouble"], out useDoubleDefine) && useDoubleDefine != 0;
                _kernel = newKernel;
            }
        }

        private long[] GlobalLaunchsizeFor(long[] localSize, long[] size)
        {
            var retval = new long[size.Length];
            for (var i = 0; i < size.Length; i++)
                retval[i] = (size[i] + localSize[i] - 1) / localSize[i] * localSize[i];
            return retval;
        }

        public long[] Threadsize(ComputeCommandQueue queue)
        {
            if (_localSize == null)
            {
                var localSizeSingle = (long)Math.Sqrt(queue.Device.MaxWorkGroupSize);
                _localSize = new long[2];
                for (var i = 0; i < 2; i++)
                    _localSize[i] = localSizeSingle;
            }
            var returnValue = new long[_localSize.Length];
            Array.Copy(_localSize, returnValue, _localSize.Length);
            return returnValue;
        }

        public void Render(ComputeBuffer<Vector4> buffer, ComputeCommandQueue queue, IParameterSet parameters, Size windowSize)
        {
            Render(buffer, queue, parameters, windowSize, 0, new Size(0, 0));
        }

        public void Render(ComputeBuffer<Vector4> buffer, ComputeCommandQueue queue, IParameterSet parameters,
            Size windowSize, int numBlocks, Size coordinates, int bufferWidth = 0)
        {
            lock (_kernelLock)
            {
                if (_kernel == null)
                    return;

                var size = new long[] { windowSize.Width, windowSize.Height };
                var localSize = Threadsize(queue);

                var offset = new long[size.Length];
                var offsetCoords = new long[] { coordinates.Width, coordinates.Height };

                if (numBlocks > 0)
                {
                    for (var i = 0; i < size.Length; i++)
                        size[i] = numBlocks * localSize[i];
                    for (var i = 0; i < size.Length; i++)
                        offset[i] = size[i] * offsetCoords[i];
                }

                var globalSize = GlobalLaunchsizeFor(localSize, size);

                var kernelNumArgs = 0;
                _kernel.SetMemoryArgument(kernelNumArgs++, buffer);
                _kernel.SetValueArgument(kernelNumArgs++, bufferWidth);
                _kernel.SetValueArgument(kernelNumArgs++, windowSize.Width);
                _kernel.SetValueArgument(kernelNumArgs++, windowSize.Height);
                parameters.ApplyToKernel(_kernel, _useDouble, ref kernelNumArgs);

                queue.Execute(_kernel, offset, globalSize, localSize, null);
            }
        }

        public void Dispose()
        {
            lock (_kernelLock)
            {
                if (_kernel != null)
                {
                    _kernel.Program.Dispose();
                    _kernel.Dispose();
                    _kernel = null;
                }
            }
        }
    }
}
