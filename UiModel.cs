using Eto.Drawing;
using Eto.Forms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Clam4
{
    internal class UiModel
    {
        private readonly UiKernel _uiKernel;
        private readonly KeyTracker _keyTracker;

        private CancellationTokenSource _cancelRun;
        private Task _cancelRunTask;
        private (int width, int height, int frames, string filename, Button goButton)? _headless;

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

        private static async Task FlushTasks(Task<ImageData>[] tasks)
        {
            for (var i = 0; i < tasks.Length; i++)
            {
                if (tasks[i] != null)
                {
                    await tasks[i];
                    tasks[i] = null;
                }
            }
        }

        private async Task Run(CancellationToken cancellation)
        {
            const int concurrentTasks = 1;
            var tasks = new Task<ImageData>[concurrentTasks];
            var oldSize = default(Size);
            while (!cancellation.IsCancellationRequested)
            {
                for (var i = 0; i < tasks.Length; i++)
                {
                    if (tasks[i] != null)
                    {
                        var image = await tasks[i];
                        _uiKernel.Display(image);
                    }
                    var newSize = UiKernel.DisplaySize;
                    if (oldSize != newSize)
                    {
                        oldSize = newSize;
                        await FlushTasks(tasks);
                        UiKernel.Kernel.Resize(newSize);
                    }
                    _keyTracker.OnRender();
                    tasks[i] = Kernel.Run(Settings, true);
                }
                if (_headless.HasValue)
                {
                    await FlushTasks(tasks);
                    var headless = _headless.Value;
                    _headless = null;
                    await RunHeadlessImpl(headless.width, headless.height, headless.frames, headless.filename, headless.goButton, cancellation);
                }
            }
        }

        public void RunHeadless(int width, int height, int frames, string filename, Button goButton)
        {
            _headless = (width, height, frames, filename, goButton);
        }

        private async Task RunHeadlessImpl(int width, int height, int frames, string filename, Button goButton, CancellationToken cancellation)
        {
            var timeEstimator = new TimeEstimator();

            void Progress(double value)
            {
                goButton.Text = $"{(value * 100):F2}% complete, {timeEstimator.Estimate(value)} left";
            };

            // should already be disabled
            goButton.Enabled = false;
            var queue = new Queue<Task<ImageData>>();
            Progress(0);
            var progressCounter = 0;
            var oldSize = Kernel.Resize(new Size(width, height));
            for (var i = 0; i < frames - 1 && !cancellation.IsCancellationRequested; i++)
            {
                queue.Enqueue(Kernel.Run(Settings, false));
                if (queue.Count > 3)
                {
                    await queue.Dequeue();
                    Progress((double)(++progressCounter) / frames);
                }
            }
            if (!cancellation.IsCancellationRequested)
            {
                queue.Enqueue(Kernel.Run(Settings, true));
            }
            while (queue.Count > (cancellation.IsCancellationRequested ? 0 : 1))
            {
                await queue.Dequeue();
                Progress((double)(++progressCounter) / frames);
            }
            if (cancellation.IsCancellationRequested)
            {
                while (queue.Count > 0)
                {
                    await queue.Dequeue();
                }
                return;
            }
            var result = await queue.Dequeue();
            Progress(1);
            var bitmap = new Bitmap(result.Width, result.Height, PixelFormat.Format32bppRgb);
            using (var locked = bitmap.Lock())
            {
                Marshal.Copy(result.Buffer, 0, locked.Data, result.Width * result.Height);
            }
            var directory = Path.GetDirectoryName(filename);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            //bitmap.Save(filename, ImageFormat.Png);
            Kernel.Resize(oldSize);
            goButton.Enabled = true;
            goButton.Text = "Go!";
        }

        public async Task Stop()
        {
            if (_cancelRun != null)
            {
                _cancelRun.Cancel();
                _cancelRun.Dispose();
                _cancelRun = null;
            }
            if (_cancelRunTask != null)
            {
                await _cancelRunTask;
                _cancelRunTask = null;
            }
        }

        public async Task<bool> StartStop()
        {
            if (IsRunning)
            {
                await Stop();
                return false;
            }
            else
            {
                _cancelRun = new CancellationTokenSource();
                _cancelRunTask = Run(_cancelRun.Token);
                return true;
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
