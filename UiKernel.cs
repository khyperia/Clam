using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Clam4
{
    internal class UiKernel : Control
    {
        private readonly Kernel _kernel;

        public UiKernel(Kernel kernel)
        {
            DoubleBuffered = true;
            this.SetStyle(ControlStyles.ContainerControl, false);
            this.SetStyle(ControlStyles.Selectable, true);
            this._kernel = kernel;
        }

        public Kernel Kernel => _kernel;

        // important! Only touch these variables on the UI thread!
        private Bitmap _bitmap;
        private ImageData _currentDisplay;

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);

            var size = this.ClientSize;
            if (size.Width > 0 && size.Height > 0)
            {
                _kernel.Resize(size);
            }
        }

        public void Display(ImageData data)
        {
            Action action = () =>
            {
                _currentDisplay.TryDispose();
                _currentDisplay = data;
                this.Invalidate();
            };
            if (this.InvokeRequired)
            {
                this.BeginInvoke(action);
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

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

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
                g.DrawImage(_bitmap, 0, 0, _bitmap.Width, _bitmap.Height);
            }
        }
    }
}