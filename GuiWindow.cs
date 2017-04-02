using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Clam4
{
    internal class GuiWindow
    {
        private readonly Form _form;
        private readonly Kernel _kernel;
        private readonly int _xOffset;
        // important! Only touch these variables on the UI thread!
        private Bitmap _bitmap;
        private ImageData _currentDisplay;

        private class DoubleBufferForm : Form
        {
            public DoubleBufferForm()
            {
                DoubleBuffered = true;
            }
        }

        public GuiWindow(Kernel kernel, int xOffset)
        {
            _kernel = kernel;
            _xOffset = xOffset;
            _form = new DoubleBufferForm()
            {
                AutoScaleMode = AutoScaleMode.None,
                Size = new Size(600 + xOffset, 400),
                Text = "Clam4",
                KeyPreview = true,
            };
            _form.Resize += OnResize;
            _form.Paint += OnPaint;
            _kernel.Resize(CorrectOffset(_form.ClientSize));
        }

        private void OnResize(object sender, EventArgs e)
        {
            var size = CorrectOffset(_form.ClientSize);
            if (size.Width > 0 && size.Height > 0)
            {
                _kernel.Resize(size);
            }
        }

        private Size CorrectOffset(Size size)
        {
            if (size.Width > _xOffset)
            {
                size.Width -= _xOffset;
            }
            else
            {
                size.Width = 0;
            }
            return size;
        }

        public void Run() => Application.Run(_form);

        public Control Form => _form;

        public void Display(ImageData data)
        {
            Action action = () =>
            {
                _currentDisplay.TryDispose();
                _currentDisplay = data;
                _form.Invalidate();
            };
            if (_form.InvokeRequired)
            {
                _form.BeginInvoke(action);
            }
            else
            {
                action();
            }
        }

        private void UpdateBitmap(ImageData currentDisplay)
        {
            if (_bitmap == null || _bitmap.Width != currentDisplay.Width || _bitmap.Height != currentDisplay.Height)
            {
                _bitmap?.Dispose();
                _bitmap = null;
                if (currentDisplay.Width > 0 && currentDisplay.Height > 0)
                {
                    _bitmap = new Bitmap(currentDisplay.Width, currentDisplay.Height, PixelFormat.Format32bppRgb);
                }
            }
            if (_bitmap != null && currentDisplay.Buffer != null)
            {
                var locked = _bitmap.LockBits(new Rectangle(new Point(0, 0), _bitmap.Size), ImageLockMode.WriteOnly,
                    _bitmap.PixelFormat);
                var lockedCount = (locked.Stride * locked.Height) / sizeof(int);
                var length = Math.Min(currentDisplay.Buffer.Length, lockedCount);
                Marshal.Copy(currentDisplay.Buffer, 0, locked.Scan0, length);
                _bitmap.UnlockBits(locked);
            }
            currentDisplay.Dispose();
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            if (_currentDisplay.Valid)
            {
                UpdateBitmap(_currentDisplay);
                _currentDisplay.Dispose();
            }
            if (_bitmap != null)
            {
                var g = e.Graphics;
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.DrawImage(_bitmap, _xOffset, 0, _bitmap.Width, _bitmap.Height);
            }
        }
    }

    internal static class UiControls
    {
        internal static readonly int UiWidth = 300;

        internal static void Create(
            Control form,
            SettingsCollection settings,
            Func<Keys, bool> onKeyDown,
            Func<Keys, bool> onKeyUp,
            Kernel kernel,
            Action<CancellationToken> run)
        {
            CancellationTokenSource cancellationTokenSource = null;

            var outerContainer = new TableLayoutPanel()
            {
                AutoScroll = false,
                Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Bottom,
                Height = form.ClientSize.Height,
                Width = UiWidth,
                ColumnCount = 1,
                RowCount = 0,
            };

            var help = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Help (keybindings, etc)",
            };
            help.Click += (o, e) =>
            {
                Task.Run(() =>
                {
                    MessageBox.Show(Program.HelpText, "Help");
                });
            };
            outerContainer.Controls.Add(help);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var startStop = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Start",
            };
            startStop.Click += (o, e) =>
            {
                if (cancellationTokenSource == null)
                {
                    cancellationTokenSource = new CancellationTokenSource();
                    run(cancellationTokenSource.Token);
                    startStop.Text = "Stop";
                }
                else
                {
                    cancellationTokenSource.Cancel();
                    cancellationTokenSource.Dispose();
                    cancellationTokenSource = null;
                    startStop.Text = "Start";
                }
            };
            outerContainer.Controls.Add(startStop);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var saveState = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Save state",
            };
            saveState.Click += (o, e) =>
            {
                using (var dialog = new SaveFileDialog()
                {
                    FileName = "settings",
                    DefaultExt = ".clam4"
                })
                {
                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        settings.Save(dialog.FileName);
                    }
                }
            };
            outerContainer.Controls.Add(saveState);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var loadState = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Load state",
            };
            loadState.Click += (o, e) =>
            {
                using (var dialog = new OpenFileDialog()
                {
                    FileName = "settings",
                    DefaultExt = ".clam4"
                })
                {
                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        settings.Load(dialog.FileName);
                    }
                }
            };
            outerContainer.Controls.Add(loadState);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var headlessRenderButton = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Run headless render",
            };
            headlessRenderButton.Click += (o, e) =>
            {
                if (cancellationTokenSource != null)
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
                        CreateHeadless(dialog.FileName, kernel.Clone(), settings.Clone());
                    }
                }
            };
            outerContainer.Controls.Add(headlessRenderButton);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var focus = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Enable keybindings",
            };
            focus.Click += (o, e) =>
            {
                focus.Focus();
            };
            form.KeyDown += (o, e) =>
            {
                if (e.KeyCode == Keys.Escape)
                {
                    e.Handled = true;
                    focus.Focus();
                    return;
                }
                if (!focus.Focused)
                {
                    return;
                }
                if (onKeyDown(e.KeyCode))
                {
                    e.Handled = true;
                }
            };
            form.KeyUp += (o, e) =>
            {
                if (!focus.Focused)
                {
                    return;
                }
                if (onKeyUp(e.KeyCode))
                {
                    e.Handled = true;
                }
            };
            outerContainer.Controls.Add(focus);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var scroll = new TableLayoutPanel()
            {
                AutoScroll = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                ColumnCount = 2,
                RowCount = 0,
            };
            scroll.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            scroll.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            foreach (var setting in settings)
            {
                var key = setting.Key;
                var isInt = setting.Value is int;
                var isFloat = setting.Value is float;
                var isDouble = setting.Value is double;
                if (!isInt && !isFloat && !isDouble)
                {
                    throw new Exception("Invalid setting type " + setting.Value.GetType().Name);
                }

                var label = new Label()
                {
                    AutoSize = true,
                    Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                    TextAlign = ContentAlignment.MiddleRight,
                    Text = key,
                };
                scroll.Controls.Add(label);

                var text = new TextBox()
                {
                    Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                    Text = setting.Value.ToString(),
                };
                var changing = false;
                settings.OnChange += (changedKey, changedValue) =>
                {
                    if (changing || text.Focused)
                    {
                        return;
                    }
                    if (changedKey == key)
                    {
                        changing = true;
                        text.Text = changedValue.ToString();
                        changing = false;
                    }
                };
                text.TextChanged += (o, e) =>
                {
                    if (changing)
                    {
                        return;
                    }
                    if (isInt)
                    {
                        if (int.TryParse(text.Text, out var result))
                        {
                            changing = true;
                            settings[key] = result;
                            changing = false;
                        }
                    }
                    if (isFloat)
                    {
                        if (float.TryParse(text.Text, out var result))
                        {
                            changing = true;
                            settings[key] = result;
                            changing = false;
                        }
                    }
                    if (isDouble)
                    {
                        if (double.TryParse(text.Text, out var result))
                        {
                            changing = true;
                            settings[key] = result;
                            changing = false;
                        }
                    }
                };
                scroll.Controls.Add(text);
            }
            outerContainer.Controls.Add(scroll);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            form.Controls.Add(outerContainer);
        }

        private static void CreateHeadless(string filename, Kernel kernel, SettingsCollection settings)
        {
            var form = new Form()
            {
                Text = "Clam4 Headless Render"
            };
            var outerContainer = new TableLayoutPanel()
            {
                AutoScroll = false,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Width = form.ClientSize.Width,
                Height = form.ClientSize.Height,
                ColumnCount = 2,
                RowCount = 0,
            };

            var filenameLabel = new Label()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                TextAlign = ContentAlignment.MiddleRight,
                Text = "Save as",
            };
            outerContainer.Controls.Add(filenameLabel);

            var filenameText = new TextBox()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = filename,
            };
            outerContainer.Controls.Add(filenameText);

            var widthLabel = new Label()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                TextAlign = ContentAlignment.MiddleRight,
                Text = "Width",
            };
            outerContainer.Controls.Add(widthLabel);

            var widthText = new TextBox()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "1000",
            };
            outerContainer.Controls.Add(widthText);

            var heightLabel = new Label()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                TextAlign = ContentAlignment.MiddleRight,
                Text = "Height",
            };
            outerContainer.Controls.Add(heightLabel);

            var heightText = new TextBox()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "1000",
            };
            outerContainer.Controls.Add(heightText);

            var framesLabel = new Label()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                TextAlign = ContentAlignment.MiddleRight,
                Text = "Frames",
            };
            outerContainer.Controls.Add(framesLabel);

            var framesText = new TextBox()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "100",
            };
            outerContainer.Controls.Add(framesText);

            var progressBar = new Label()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "",
            };
            outerContainer.Controls.Add(progressBar);
            outerContainer.SetColumnSpan(progressBar, 2);

            var goButton = new Button()
            {
                AutoSize = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Go!",
            };
            outerContainer.Controls.Add(goButton);
            outerContainer.SetColumnSpan(goButton, 2);
            TimeEstimator timeEstimator;
            void Progress(double value)
            {
                progressBar.Text = $"{(value * 100):F2}% complete, {timeEstimator.Estimate(value)} left";
            };

            var cancel = false;
            goButton.Click += async (o, e) =>
            {
                if (int.TryParse(widthText.Text, out var width) &&
                    int.TryParse(heightText.Text, out var height) &&
                    int.TryParse(framesText.Text, out var frames))
                {
                    goButton.Enabled = false;
                    var queue = new Queue<Task<ImageData>>();
                    timeEstimator = new TimeEstimator();
                    Progress(0);
                    int progressCounter = 0;
                    kernel.Resize(new Size(width, height));
                    for (var i = 0; i < frames - 1 && !cancel; i++)
                    {
                        queue.Enqueue(kernel.Run(settings, false));
                        if (queue.Count > 3)
                        {
                            await queue.Dequeue();
                            Progress((double)(++progressCounter) / frames);
                        }
                    }
                    if (!cancel)
                    {
                        queue.Enqueue(kernel.Run(settings, true));
                    }
                    while (queue.Count > (cancel ? 0 : 1))
                    {
                        await queue.Dequeue();
                        Progress((double)(++progressCounter) / frames);
                    }
                    if (cancel)
                    {
                        return;
                    }
                    var result = await queue.Dequeue();
                    Progress(1);
                    var bitmap = new Bitmap(result.Width, result.Height, PixelFormat.Format32bppRgb);
                    var locked = bitmap.LockBits(new Rectangle(new Point(0, 0), bitmap.Size), ImageLockMode.WriteOnly, bitmap.PixelFormat);
                    var lockedCount = (locked.Stride * locked.Height) / sizeof(int);
                    var length = Math.Min(result.Buffer.Length, lockedCount);
                    Marshal.Copy(result.Buffer, 0, locked.Scan0, length);
                    bitmap.UnlockBits(locked);
                    bitmap.Save(filename);
                    progressBar.Text = "Finished";
                }
            };

            form.FormClosing += (o, e) =>
            {
                cancel = true;
            };

            form.Controls.Add(outerContainer);

            form.Show();
        }
    }
}