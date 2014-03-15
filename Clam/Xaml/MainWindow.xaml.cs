using System;
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

namespace Clam.Xaml
{
    public partial class MainWindow : INotifyPropertyChanged
    {
        [CanBeNull]
        private RenderWindow _renderWindow;
        private string _selectedKernelXmlFile;
        public ObservableCollection<ComputeDevice> Devices { get; set; }
        public ObservableCollection<string> KernelXmlFiles { get; set; }
        public ObservableCollection<DefineViewModel> DefineViewModels { get; set; }

        public MainWindow()
        {
            KernelXmlFiles = new ObservableCollection<string>();
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
            foreach (var xmlFile in Directory.EnumerateFiles(Environment.CurrentDirectory, "*.xml"))
                KernelXmlFiles.Add(Path.GetFileName(xmlFile));
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

        public string SelectedKernelXmlFile
        {
            get { return _selectedKernelXmlFile; }
            set
            {
                if (value == _selectedKernelXmlFile || _renderWindow == null) return;
                _selectedKernelXmlFile = value;
                OnPropertyChanged();
                _renderWindow.Invoke(() =>
                {
                    var package = RenderPackage.LoadFromXml(_renderWindow.ComputeContext, value);
                    if (package.HasValue == false)
                        return;
                    Dispatcher.BeginInvoke(new Action(() => RenderPackage = package.Value));
                });
            }
        }

        public RenderPackage RenderPackage
        {
            get { return _renderWindow == null ? default(RenderPackage) : _renderWindow.Renderer; }
            set
            {
                if (_renderWindow == null || value.Equals(_renderWindow.Renderer)) return;
                OnPropertyChanged();
                _renderWindow.Renderer = value;
                OnPropertyChanged("ControlsHelp");
                RefreshDefines();
            }
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
            if (_renderWindow == null)
                return;
            _renderWindow.Renderer.Kernel.Recompile();
            var framedepcontrols = _renderWindow.Renderer.Parameters as IFrameDependantControl;
            if (framedepcontrols != null)
                framedepcontrols.Frame = 0;
        }

        private void LoadKernelParameters(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null)
                return;
            var selector = new OpenFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == true)
                _renderWindow.Renderer.Kernel.SerializeOptions().Save(selector.FileName);
        }

        private void SaveKernelParameters(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null)
                return;
            var selector = new SaveFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                DefaultExt = ".kernel.xml",
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == true)
                _renderWindow.Renderer.Kernel.LoadOptions(XDocument.Load(selector.FileName).Root);
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
            if (_renderWindow == null)
                return;
            ThreadPool.QueueUserWorkItem(o => _renderWindow.Renderer.
                Screenshot(StaticSettings.Fetch.ScreenshotHeight, StaticSettings.Fetch.ScreenshotPartialRender, _renderWindow.DisplayInformation)
                .Save(Ext.UniqueFilename("screenshot", "png")));
        }

        private void TakeGif(object sender, RoutedEventArgs e)
        {
            if (_renderWindow == null)
                return;
            var gifable = _renderWindow.Renderer.Parameters as IGifableControl;
            if (gifable != null)
                ThreadPool.QueueUserWorkItem(o => _renderWindow.Renderer.TakeGif(gifable, _renderWindow.DisplayInformation));
        }

        private void SaveKernelState(object sender, RoutedEventArgs ev)
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

        private void LoadKernelState(object sender, RoutedEventArgs e)
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
            serializable.Save().Save(filename);
            _renderWindow.DisplayInformation("Saved state");
        }
    }
}
