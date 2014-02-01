using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text.RegularExpressions;
using Cloo;
using JetBrains.Annotations;
using OpenTK;

namespace Clam
{
    interface IParameterSet
    {
        void ApplyToKernel(ComputeKernel kernel, ref int startIndex);
        void UnRegister();
    }

    class RenderKernel : IDisposable
    {
        private readonly string[] _sourcecodes;
        private readonly Dictionary<string, string> _defines;
        private ComputeKernel _kernel;
        private long[] _localSize;

        private RenderKernel(ComputeKernel kernel, string[] sourcecodes, Dictionary<string, string> defines)
        {
            _kernel = kernel;
            _sourcecodes = sourcecodes;
            _defines = defines;
        }

        public ComputeContext ComputeContext { get { return _kernel.Context; } }

        private static ComputeKernel Compile(ComputeContext context, string[] sourcecodes, Dictionary<string, string> defines)
        {
            var program = new ComputeProgram(context, sourcecodes);
            var device = context.Devices.Single();
            try
            {
                foreach (var define in defines.Where(define => define.Key.Any(char.IsWhiteSpace) || define.Value.Any(char.IsWhiteSpace)))
                {
                    Console.WriteLine("Invalid define \"" + define.Key + "=" + define.Value + "\": define contained whitespace");
                    return null;
                }
                var options = string.Join(" ", defines.Where(kvp => !string.IsNullOrEmpty(kvp.Value)).Select(kvp => "-D " + kvp.Key + "=" + kvp.Value));
                program.Build(new[] { device }, options, null, IntPtr.Zero);
                var str = program.GetBuildLog(device).Trim();
                if (string.IsNullOrEmpty(str) == false)
                    Console.WriteLine(str);
                return program.CreateKernel("Main");
            }
            catch (InvalidBinaryComputeException)
            {
                Console.WriteLine(program.GetBuildLog(device));
                return null;
            }
            catch (BuildProgramFailureComputeException)
            {
                Console.WriteLine(program.GetBuildLog(device));
                return null;
            }
        }

        private static readonly Regex DefineRegex = new Regex(@"^#ifndef +(\w+)\r?$", RegexOptions.Multiline);
        private static IEnumerable<string> CollectDefines(IEnumerable<string> sourcecodes)
        {
            return sourcecodes.SelectMany(sourcecode => DefineRegex.Matches(sourcecode).Cast<Match>(), (sourcecode, match) => match.Groups[1].Captures[0].Value);
        }

        [CanBeNull]
        public static RenderKernel Create(ComputeContext context, string[] sourcecodes)
        {
            var defines = CollectDefines(sourcecodes).ToDictionary(define => define, define => "");
            var compilation = Compile(context, sourcecodes, defines);
            return compilation == null ? null : new RenderKernel(compilation, sourcecodes, defines);
        }

        public IEnumerable<KeyValuePair<string, string>> Options
        {
            get { return _defines; }
        }

        public void SetOption(string key, string value)
        {
            if (_defines.ContainsKey(key) == false)
                throw new Exception("Define " + key + " does not exist, while trying to set it's value in RenderKernel");
            _defines[key] = value;
        }

        public void Recompile()
        {
            var context = _kernel.Context;
            if (_kernel != null)
            {
                lock (_kernel)
                {
                    _kernel.Dispose();
                    _kernel = null;
                }
            }
            _kernel = Compile(context, _sourcecodes, _defines);
        }

        private long[] GlobalLaunchsizeFor(long[] size)
        {
            var retval = new long[size.Length];
            for (var i = 0; i < size.Length; i++)
                retval[i] = (size[i] + _localSize[i] - 1) / _localSize[i] * _localSize[i];
            return retval;
        }

        public void Render(ComputeBuffer<Vector4> buffer, ComputeCommandQueue queue, IParameterSet parameters, Size windowSize)
        {
            Render(buffer, queue, parameters, windowSize, 1, new Size(0, 0));
        }

        public void Render(ComputeBuffer<Vector4> buffer, ComputeCommandQueue queue, IParameterSet parameters, Size windowSize, int partialRender, Size coordinates)
        {
            if (_kernel == null)
                return;
            lock (_kernel)
            {
                _kernel.SetMemoryArgument(0, buffer);
                _kernel.SetValueArgument(1, windowSize.Width);
                _kernel.SetValueArgument(2, windowSize.Height);
                var kernelNumArgs = 3;
                parameters.ApplyToKernel(_kernel, ref kernelNumArgs);

                var size = new long[] { windowSize.Width, windowSize.Height };
                if (_localSize == null)
                {
                    var localSizeSingle = (long)Math.Pow(queue.Device.MaxWorkGroupSize, 1.0 / size.Length);
                    _localSize = new long[size.Length];
                    for (var i = 0; i < size.Length; i++)
                        _localSize[i] = localSizeSingle;
                }
                var offset = new long[size.Length];
                var offsetCoords = new long[] { coordinates.Width, coordinates.Height };

                if (partialRender > 1)
                {
                    for (var i = 0; i < size.Length; i++)
                        size[i] /= partialRender;
                    for (var i = 0; i < size.Length; i++)
                        offset[i] = size[i] * offsetCoords[i];
                }

                var globalSize = GlobalLaunchsizeFor(size);
                queue.Execute(_kernel, offset, globalSize, _localSize, null);
            }
        }

        public void Dispose()
        {
            if (_kernel != null)
                lock (_kernel)
                {
                    _kernel.Program.Dispose();
                    _kernel.Dispose();
                    _kernel = null;
                }
        }
    }
}
