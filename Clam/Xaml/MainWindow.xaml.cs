using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Windows;
using System.Xml.Linq;
using Cloo;
using JetBrains.Annotations;
using Microsoft.Win32;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Clam.Xaml
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

    public partial class MainWindow : INotifyPropertyChanged
    {
        [CanBeNull]
        private RenderWindow _renderWindow;
        private KernelXmlFile _selectedKernelXmlFile;
        public ObservableCollection<ComputeDevice> Devices { get; set; }
        public ObservableCollection<KernelXmlFile> KernelXmlFiles { get; set; }
        public ObservableCollection<DefineViewModel> DefineViewModels { get; set; }

        public MainWindow()
        {
            KernelXmlFiles = new ObservableCollection<KernelXmlFile>();
            DefineViewModels = new ObservableCollection<DefineViewModel>();
            RefreshXmlFiles();
            Devices = new ObservableCollection<ComputeDevice>(ComputePlatform.Platforms.SelectMany(p => p.Devices));
            if (Devices.Count == 0)
            {
                MessageBox.Show(this, "No OpenCL-compatible GPUs found", "Error");
            }
            InitializeComponent();
            if (Devices.Count == 1)
                SelectedDevice = Devices[0];
        }

        private void RefreshXmlFiles()
        {
            KernelXmlFiles.Clear();
            foreach (var xmlFile in Directory.EnumerateFiles(Environment.CurrentDirectory, "*.xml")
                .SelectMany(xmlFile => KernelXmlFile.Load(XElement.Load(xmlFile))))
                KernelXmlFiles.Add(xmlFile);
        }

        private void RefreshDefines()
        {
            if (_renderWindow == null)
                return;
            DefineViewModels.Clear();
            if (_renderWindow.Renderer.Kernel == null)
                return;
            foreach (var option in _renderWindow.Renderer.Kernel.Options)
                DefineViewModels.Add(new DefineViewModel(option.Key, option.Value, _renderWindow.Renderer.Kernel));
        }

        public Visibility DeviceSelectedVisibility
        {
            get { return _renderWindow != null ? Visibility.Visible : Visibility.Collapsed; }
        }

        public Visibility DeviceNotSelectedVisibility
        {
            get { return _renderWindow == null ? Visibility.Visible : Visibility.Collapsed; }
        }

        public ComputeDevice SelectedDevice
        {
            set
            {
                if (_renderWindow != null)
                    _renderWindow.Dispose();
                _renderWindow = new RenderWindow(value, s => StatusBarTextBlock.Text = s);
                WindowsFormsHost.KeyDown += (sender, args) => args.Handled = true;
                WindowsFormsHost.Child = _renderWindow;
                OnPropertyChanged("DeviceSelectedVisibility");
                OnPropertyChanged("DeviceNotSelectedVisibility");
            }
        }

        public KernelXmlFile SelectedKernelXmlFile
        {
            get { return _selectedKernelXmlFile; }
            set
            {
                if (value == _selectedKernelXmlFile || _renderWindow == null) return;
                var rightclick = System.Windows.Input.Mouse.RightButton == System.Windows.Input.MouseButtonState.Pressed;
                SetSelectedKernelXmlFile(value, rightclick, null);
            }
        }

        private void SetSelectedKernelXmlFile(KernelXmlFile filename, bool keepControl, Action callback)
        {
            _selectedKernelXmlFile = filename;
            OnPropertyChanged("SelectedKernelXmlFile");
            if (_renderWindow == null)
                return;
            _renderWindow.BeginInvoke((Action)(() =>
            {
                var package = RenderPackage.LoadFromXml(_renderWindow.ComputeContext, filename);
                if (package.HasValue == false)
                    return;
                Dispatcher.BeginInvoke(new Action(() =>
                {
                    _renderWindow.Renderer = keepControl ? new RenderPackage(package.Value.Kernel, _renderWindow.Renderer.Parameters) : package.Value;
                    var ifdp = _renderWindow.Renderer.Parameters as IFrameDependantControl;
                    if (keepControl && ifdp != null)
                    {
                        ifdp.Frame = 0;
                    }
                    OnPropertyChanged("ControlsHelp");
                    RefreshDefines();
                    if (callback != null)
                        callback();
                }));
            }));
        }

        public string ControlsHelp
        {
            get
            {
                if (_renderWindow == null)
                    return "";
                var parameters = _renderWindow.Renderer.Parameters;
                if (parameters == null)
                    return "";
                return parameters.ControlsHelp;
            }
        }

        public StaticSettings StaticSettings
        {
            get { return StaticSettings.Fetch; }
        }

        private void ApplyKernelParameters(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null || _renderWindow.Renderer.Kernel == null)
                return;
            _renderWindow.Renderer.Kernel.Recompile();
            var framedepcontrols = _renderWindow.Renderer.Parameters as IFrameDependantControl;
            if (framedepcontrols != null)
                framedepcontrols.Frame = 0;
        }

        private void LoadKernelParameters(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null || _renderWindow.Renderer.Kernel == null)
                return;
            var selector = new OpenFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == true)
                _renderWindow.Renderer.Kernel.LoadOptions(XDocument.Load(selector.FileName).Root);
        }

        private void SaveKernelParameters(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null || _renderWindow.Renderer.Kernel == null)
                return;
            var selector = new SaveFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                DefaultExt = ".kernel.xml",
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == true)
                _renderWindow.Renderer.Kernel.SerializeOptions("KernelOptions").Save(selector.FileName);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            var handler = PropertyChanged;
            if (handler != null)
                handler(this, new PropertyChangedEventArgs(propertyName));
        }

        public class DefineViewModel : INotifyPropertyChanged
        {
            private readonly RenderKernel _renderKernel;
            private string _name;
            private string _value;

            public string Name
            {
                get { return _name; }
                set
                {
                    if (value == _name) return;
                    _name = value;
                    OnPropertyChanged1();
                }
            }

            public string Value
            {
                get { return _value; }
                set
                {
                    if (value == _value) return;
                    _value = value;
                    OnPropertyChanged1();
                    _renderKernel.SetOption(_name, _value);
                }
            }

            public DefineViewModel(string name, string value, RenderKernel renderKernel)
            {
                _renderKernel = renderKernel;
                _name = name;
                _value = value;
            }

            public event PropertyChangedEventHandler PropertyChanged;

            [NotifyPropertyChangedInvocator]
            protected virtual void OnPropertyChanged1([CallerMemberName] string propertyName = null)
            {
                var handler = PropertyChanged;
                if (handler != null) handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        private void TakeScreenshot(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null || _renderWindow.Renderer.Parameters == null)
                return;
            ThreadPool.QueueUserWorkItem(o =>
            {
                var endIgnoreControl = _renderWindow.Renderer.Parameters.StartIgnoreControl();
                var screenshot = _renderWindow.Renderer.Screenshot(StaticSettings.Fetch.ScreenshotHeight,
                    StaticSettings.Fetch.ScreenshotPartialRender, _renderWindow.DisplayInformation);
                _renderWindow.DisplayInformation("Saving screenshot");
                if (screenshot != null)
                    screenshot.Save(Ext.UniqueFilename("screenshot", "png"));
                _renderWindow.DisplayInformation("Done rendering screenshot");
                endIgnoreControl();
            });
        }

        private void TakeGif(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null)
                return;
            var gifable = _renderWindow.Renderer.Parameters as IGifableControl;
            if (gifable != null)
                ThreadPool.QueueUserWorkItem(o =>
                {
                    var endIgnoreControl = _renderWindow.Renderer.Parameters.StartIgnoreControl();
                    _renderWindow.Renderer.TakeGif(gifable, _renderWindow.DisplayInformation);
                    endIgnoreControl();
                });
        }

        private void SaveControlState(object sender, RoutedEventArgs ev)
        {
            if (_renderWindow == null)
                return;
            var serializable = _renderWindow.Renderer.Parameters as ISerializableParameterSet;
            if (serializable == null)
            {
                MessageBox.Show(this, "Loading is not supported with this kernel", "Error");
                return;
            }
            var filename = Path.Combine("..", "State", GetType().Name + ".state.xml");
            var directory = Path.GetDirectoryName(filename);
            if (directory != null && Directory.Exists(directory) == false)
                Directory.CreateDirectory(directory);
            serializable.Save("ControlState").Save(filename);
            _renderWindow.DisplayInformation("Saved state");
        }

        private void LoadControlState(object sender, RoutedEventArgs ev)
        {
            if (_renderWindow == null)
                return;
            var serializable = _renderWindow.Renderer.Parameters as ISerializableParameterSet;
            if (serializable == null)
            {
                MessageBox.Show(this, "Saving is not supported with this kernel", "Error");
                return;
            }
            try
            {
                serializable.Load(XDocument.Load(Path.Combine("..", "State", GetType().Name + ".state.xml")).Root);
                var fdc = _renderWindow.Renderer.Parameters as IFrameDependantControl;
                if (fdc != null)
                    fdc.Frame = 0;
                _renderWindow.DisplayInformation("Loaded state");
            }
            catch (Exception e)
            {
                _renderWindow.DisplayInformation("Could not load state: " + e.GetType().Name + ": " + e.Message);
            }
        }

        private void ExportConfig(object sender, RoutedEventArgs e)
        {
            var selector = new SaveFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                DefaultExt = ".clam.xml",
                Filter = "Xml Clam files|*.clam.xml"
            };
            if (selector.ShowDialog(this) == true)
                SaveProgram(selector.FileName);
        }

        private void ImportConfig(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null)
                return;
            var selector = new OpenFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                Filter = "Xml Clam files|*.clam.xml"
            };
            if (selector.ShowDialog(this) == true)
                LoadProgram(selector.FileName);
        }

        private void SaveProgram(string filename)
        {
            if (_renderWindow == null || _renderWindow.Renderer.Kernel == null)
                return;
            var kernel = SelectedKernelXmlFile;
            var kernelOptions = _renderWindow.Renderer.Kernel.SerializeOptions("KernelOptions");
            var serializableParams = _renderWindow.Renderer.Parameters as ISerializableParameterSet;
            var controlOptions = serializableParams == null ? null : serializableParams.Save("ControlOptions");
            var element = new XElement("ClamConfig", kernel.Save("Kernel"), kernelOptions);
            if (controlOptions != null)
                element.Add(controlOptions);
            element.Save(filename);
        }

        private void LoadProgram(string filename)
        {
            if (_renderWindow == null)
                return;
            var doc = XElement.Load(filename);
            if (doc.Name.LocalName != "ClamConfig")
            {
                MessageBox.Show(this, "XML was not a ClamConfig file", "Error");
                return;
            }
            var kernelElement = doc.Element("Kernel");
            var kernelOptionsElement = doc.Element("KernelOptions");
            if (kernelElement == null || kernelOptionsElement == null)
            {
                MessageBox.Show(this, "XML settings are incomplete", "Error");
                return;
            }
            SetSelectedKernelXmlFile(KernelXmlFile.Load(kernelElement).Single(), false, () =>
            {
                if (_renderWindow.Renderer.Kernel == null)
                {
                    MessageBox.Show("Internal error; renderer not set when loading program", "Error");
                    return;
                }
                _renderWindow.Renderer.Kernel.LoadOptions(kernelOptionsElement);
                RefreshDefines();
                var serializable = _renderWindow.Renderer.Parameters as ISerializableParameterSet;
                if (serializable != null)
                {
                    var element = doc.Element("ControlOptions");
                    if (element == null)
                        MessageBox.Show(this, "XML settings are incomplete: missing ControlOptions", "Warning");
                    else
                        serializable.Load(element);
                }
            });
        }
    }
}
