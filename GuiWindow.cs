using System;
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

        private void OnResize(object sender, EventArgs e) => _kernel.Resize(CorrectOffset(_form.ClientSize));

        private Size CorrectOffset(Size size)
        {
            size.Width -= _xOffset;
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

        internal static void Create(Control form, SettingsCollection settings, Func<Keys, bool> onKeyDown, Func<Keys, bool> onKeyUp, Action<CancellationToken> run)
        {
            CancellationTokenSource cancellationTokenSource = null;
            var uiWidthMinusScroll = UiWidth - 20;
            var ySpace = 1;
            var yPos = ySpace + 5;
            var yStride = 20;
            var yHeight = yStride - ySpace * 2;
            var yIdx = 0;
            var xSpace = 5;
            var widthFull = uiWidthMinusScroll - xSpace * 2;
            var xPos = xSpace;
            var widthHalf = uiWidthMinusScroll / 2 - xSpace * 2;
            var xPosHalf = uiWidthMinusScroll / 2 + xSpace;

            var help = new Button()
            {
                Left = xPos,
                Top = yPos + yStride * yIdx++,
                Width = widthFull,
                Height = yHeight,
                Text = "Help (keybindings, etc)",
            };
            help.Click += (o, e) =>
            {
                Task.Run(() =>
                {
                    MessageBox.Show(Program.HelpText, "Help");
                });
            };
            form.Controls.Add(help);

            var startStop = new Button()
            {
                Left = xPos,
                Top = yPos + yStride * yIdx++,
                Width = widthFull,
                Height = yHeight,
                Text = "start/stop",
            };
            startStop.Click += (o, e) =>
            {
                if (cancellationTokenSource == null)
                {
                    cancellationTokenSource = new CancellationTokenSource();
                    run(cancellationTokenSource.Token);
                }
                else
                {
                    cancellationTokenSource.Cancel();
                    cancellationTokenSource.Dispose();
                    cancellationTokenSource = null;
                }
            };
            form.Controls.Add(startStop);

            var focus = new Button()
            {
                Left = xPos,
                Top = yPos + yStride * yIdx++,
                Width = widthFull,
                Height = yHeight,
                Text = "Focus keybindings",
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
            form.Controls.Add(focus);

            var scroll = new Panel()
            {
                AutoScroll = true,
                Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Bottom,
                Left = 0,
                Top = yPos + yStride * yIdx++,
                Width = UiWidth,
            };
            scroll.Height = form.ClientSize.Height - scroll.Top;

            yIdx = 0;
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
                    Left = xPos,
                    Top = yPos + yStride * yIdx,
                    Width = widthHalf,
                    Height = yHeight,
                    Text = key,
                };
                scroll.Controls.Add(label);

                var text = new TextBox()
                {
                    Left = xPosHalf,
                    Top = yPos + yStride * yIdx++,
                    Width = widthHalf,
                    Height = yHeight,
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
            form.Controls.Add(scroll);
        }
    }
}