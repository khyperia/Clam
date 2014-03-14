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
        private RenderWindow _renderWindow;
        private string _selectedKernelXmlFile;
        public ObservableCollection<string> KernelXmlFiles { get; set; }
        public ObservableCollection<DefineViewModel> DefineViewModels { get; set; }

        public MainWindow()
        {
            KernelXmlFiles = new ObservableCollection<string>();
            DefineViewModels = new ObservableCollection<DefineViewModel>();
            RefreshXmlFiles();
            InitializeComponent();
            var autoResetEvent = new AutoResetEvent(false);
            new Thread(BootWindow).Start(autoResetEvent);
            autoResetEvent.WaitOne();
        }

        private void BootWindow(object syncObj)
        {
            var device = ComputePlatform.Platforms.SelectMany(p => p.Devices).Single(); // TODO
            _renderWindow = new RenderWindow(device);
            ((AutoResetEvent)syncObj).Set();
            _renderWindow.Run();
            _renderWindow.Dispose();
            _renderWindow = null;
        }

        private void RefreshXmlFiles()
        {
            KernelXmlFiles.Clear();
            foreach (var xmlFile in Directory.EnumerateFiles(Environment.CurrentDirectory, "*.xml"))
                KernelXmlFiles.Add(Path.GetFileName(xmlFile));
        }

        private void RefreshDefines()
        {
            DefineViewModels.Clear();
            if (_renderWindow.Renderer.Kernel == null)
                return;
            foreach (var option in _renderWindow.Renderer.Kernel.Options)
                DefineViewModels.Add(new DefineViewModel(option.Key, option.Value, _renderWindow.Renderer.Kernel));
        }

        public string SelectedKernelXmlFile
        {
            get { return _selectedKernelXmlFile; }
            set
            {
                if (value == _selectedKernelXmlFile) return;
                _selectedKernelXmlFile = value;
                OnPropertyChanged();
                _renderWindow.Invoke(() =>
                {
                    var package = RenderPackage.LoadFromXml(_renderWindow, value);
                    if (package.HasValue == false)
                        return;
                    Dispatcher.BeginInvoke(new Action(() => RenderPackage = package.Value));
                });
            }
        }

        public RenderPackage RenderPackage
        {
            get { return _renderWindow.Renderer; }
            set
            {
                if (value.Equals(_renderWindow.Renderer)) return;
                OnPropertyChanged();
                _renderWindow.Renderer = value;
                RefreshDefines();
            }
        }

        public StaticSettings StaticSettings
        {
            get { return StaticSettings.Fetch; }
        }

        private void ApplyKernelParameters(object sender, RoutedEventArgs e)
        {
            _renderWindow.Renderer.Kernel.Recompile();
            var framedepcontrols = _renderWindow.Renderer.Parameters as IFrameDependantControl;
            if (framedepcontrols != null)
                framedepcontrols.Frame = 0;
        }

        private void LoadKernelParameters(object sender, RoutedEventArgs e)
        {
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
            var selector = new SaveFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                DefaultExt = ".kernel.xml",
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == true)
                _renderWindow.Renderer.Kernel.LoadOptions(XDocument.Load(selector.FileName).Root);
        }

        protected override void OnClosed(EventArgs e)
        {
            _renderWindow.CancelClosing = false;
            _renderWindow.Invoke(_renderWindow.Exit);
            base.OnClosed(e);
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
    }
}
