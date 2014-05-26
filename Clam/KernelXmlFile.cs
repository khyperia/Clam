using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;

namespace Clam
{
    public class KernelXmlFile
    {
        private readonly string _controlsName;
        private readonly string _name;
        private readonly Func<IParameterSet> _controlsFunc;
        private readonly string[] _files;

        private KernelXmlFile(string name, string controlsName, Func<IParameterSet> controlsFunc, string[] files)
        {
            _name = name;
            _controlsName = controlsName;
            _controlsFunc = controlsFunc;
            _files = files;
        }

        public string Name
        {
            get { return _name; }
        }

        public override string ToString()
        {
            return _name;
        }

        public Func<IParameterSet> ControlsFunc
        {
            get { return _controlsFunc; }
        }

        public string[] Files
        {
            get { return _files; }
        }

        public static IEnumerable<KernelXmlFile> Load(XElement xml)
        {
            if (xml == null)
                throw new Exception("Invalid XML file: no root element");
            var controls = xml.Element("Controls");
            if (controls == null)
                throw new Exception("Invalid XML file: no Controls element");
            if (!RenderPackage.ControlBindingNames.ContainsKey(controls.Value))
                throw new Exception("Invalid XML file: Controls value did not exist, possible values are: " + string.Join(", ", RenderPackage.ControlBindingNames));
            var controlsFunc = RenderPackage.ControlBindingNames[controls.Value];
            var files = xml.Element("Files");
            if (files == null)
                throw new Exception("Invalid XML file: no Files element");
            var sourceFiles = files.Elements("File").Select(f => f.Value).ToArray();
            var optionFiles = CartesianProduct(files.Elements("FileChoice").Select(e => e.Elements("File").Select(el => el.Value)));
            return optionFiles.Select(optionFile =>
            {
                var realSources = sourceFiles.Concat(optionFile).ToArray();
                var name = xml.Name.LocalName;
                for (var i = sourceFiles.Length; i < realSources.Length; i++)
                    name += string.Format("[{0}]", realSources[i].EndsWith(".cl") ? realSources[i].Substring(0, realSources[i].Length - 3) : realSources[i]);
                return new KernelXmlFile(name, controls.Value, controlsFunc, realSources);
            });
        }

        public XElement Save(string name)
        {
            return new XElement(name,
                new XElement("Controls", _controlsName),
                new XElement("Files", _files.Select(s => new XElement("File", s))));
        }

        private static IEnumerable<IEnumerable<T>> CartesianProduct<T>(IEnumerable<IEnumerable<T>> sequences)
        {
            IEnumerable<IEnumerable<T>> emptyProduct =
                new[] { Enumerable.Empty<T>() };
            return sequences.Aggregate(
                emptyProduct,
                (accumulator, sequence) =>
                    from accseq in accumulator
                    from item in sequence
                    select accseq.Concat(new[] { item }));
        }
    }
}