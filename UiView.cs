using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Clam4
{
    internal static class UiView
    {
        internal static readonly int UiWidth = 300;

        internal static Form Create(UiModel model)
        {
            var form = new Form()
            {
                Text = "Clam4",
                Width = 600 + UiWidth,
                Height = 400,
                KeyPreview = true,
            };

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
            help.Click += (o, e) => model.Help();
            outerContainer.Controls.Add(help);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var startStop = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Start",
            };
            startStop.Click += (o, e) =>
            {
                if (model.StartStop())
                {
                    startStop.Text = "Stop";
                }
                else
                {
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
            saveState.Click += (o, e) => model.SaveState();
            outerContainer.Controls.Add(saveState);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var loadState = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Load state",
            };
            loadState.Click += (o, e) => model.LoadState();
            outerContainer.Controls.Add(loadState);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var headlessRenderButton = new Button()
            {
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                Text = "Run headless render",
            };
            headlessRenderButton.Click += (o, e) => model.RunHeadless();
            outerContainer.Controls.Add(headlessRenderButton);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            model.UiKernel.Click += (o, e) => model.UiKernel.Select();
            form.KeyDown += (o, e) =>
            {
                if (e.KeyCode == Keys.Escape)
                {
                    e.Handled = true;
                    model.UiKernel.Select();
                    return;
                }
            };
            model.UiKernel.PreviewKeyDown += (o, e) =>
            {
                var code = e.KeyCode;
                var isArrow = code == Keys.Up || code == Keys.Down || code == Keys.Left || code == Keys.Right;
                if (isArrow)
                {
                    e.IsInputKey = true;
                }
            };
            model.UiKernel.KeyDown += (o, e) =>
            {
                if (model.OnKeyDown(e.KeyCode))
                {
                    e.Handled = true;
                }
            };
            model.UiKernel.KeyUp += (o, e) =>
            {
                if (model.OnKeyUp(e.KeyCode))
                {
                    e.Handled = true;
                }
            };

            var scroll = new TableLayoutPanel()
            {
                AutoScroll = true,
                Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                ColumnCount = 2,
                RowCount = 0,
            };
            scroll.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            scroll.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            foreach (var setting in model.Settings)
            {
                var key = setting.Key;

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
                model.Settings.OnChange += (changedKey, changedValue) =>
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
                    model.SetSetting(key, text.Text);
                };
                scroll.Controls.Add(text);
            }
            outerContainer.Controls.Add(scroll);
            outerContainer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            form.Controls.Add(outerContainer);

            model.UiKernel.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            model.UiKernel.Top = 0;
            model.UiKernel.Height = form.ClientSize.Height;
            model.UiKernel.Left = UiWidth;
            model.UiKernel.Width = form.ClientSize.Width - UiWidth;
            form.Controls.Add(model.UiKernel);

            return form;
        }

        public static void CreateHeadless(string filename, Kernel kernel, SettingsCollection settings)
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