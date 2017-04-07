using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Clam4
{
    internal class UiModel
    {
        private readonly UiKernel _uiKernel;
        private readonly KeyTracker _keyTracker;

        private CancellationTokenSource _cancelRun;

        public UiModel(UiKernel uiKernel, KeyTracker keyTracker)
        {
            _uiKernel = uiKernel;
            _keyTracker = keyTracker;
        }

        public SettingsCollection Settings => _keyTracker.Settings;

        public UiKernel UiKernel => _uiKernel;

        private Kernel Kernel => _uiKernel.Kernel;

        public bool OnKeyDown(Keys keys) => _keyTracker.OnKeyDown(keys);

        public bool OnKeyUp(Keys keys) => _keyTracker.OnKeyUp(keys);

        public bool IsRunning => _cancelRun != null;

        private async void Run(CancellationToken cancellation)
        {
            const int concurrentTasks = 1;
            var tasks = new Task<ImageData>[concurrentTasks];
            while (!cancellation.IsCancellationRequested)
            {
                for (var i = 0; i < tasks.Length; i++)
                {
                    if (tasks[i] != null)
                    {
                        var image = await tasks[i];
                        _uiKernel.Display(image);
                    }
                    _keyTracker.OnRender();
                    tasks[i] = Kernel.Run(Settings, true);
                }
            }
        }

        public void Help()
        {
            Task.Run(() =>
            {
                MessageBox.Show(Program.HelpText, "Help");
            });
        }

        public bool StartStop()
        {
            if (IsRunning)
            {
                _cancelRun.Cancel();
                _cancelRun.Dispose();
                _cancelRun = null;
                return false;
            }
            else
            {
                _cancelRun = new CancellationTokenSource();
                Run(_cancelRun.Token);
                return true;
            }
        }

        public void RunHeadless()
        {
            if (_cancelRun != null)
            {
                MessageBox.Show("Do not start a headless render while the active render is running (press Stop)", "Error");
                return;
            }
            using (var dialog = new SaveFileDialog()
            {
                FileName = "render",
                DefaultExt = ".png"
            })
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    UiView.CreateHeadless(dialog.FileName, Kernel.Clone(), Settings.Clone());
                }
            }
        }

        public void SaveState()
        {
            using (var dialog = new SaveFileDialog()
            {
                FileName = "settings",
                DefaultExt = ".clam4"
            })
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    Settings.Save(dialog.FileName);
                }
            }
        }

        public void LoadState()
        {
            using (var dialog = new OpenFileDialog()
            {
                FileName = "settings",
                DefaultExt = ".clam4"
            })
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    Settings.Load(dialog.FileName);
                }
            }
        }

        public void SetSetting(string key, string value)
        {
            var old = Settings[key];
            if (old is int)
            {
                if (int.TryParse(value, out var result))
                {
                    Settings[key] = result;
                }
            }
            else if (old is float)
            {
                if (float.TryParse(value, out var result))
                {
                    Settings[key] = result;
                }
            }
            else if (old is double)
            {
                if (double.TryParse(value, out var result))
                {
                    Settings[key] = result;
                }
            }
            else
            {
                throw new Exception("Invalid setting type " + old.GetType().Name);
            }
        }
    }
}