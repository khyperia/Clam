using System;
using System.Runtime.InteropServices;
using Eto.Forms;
using Eto.Drawing;

namespace Clam4
{
    internal class UiKernel
    {
        private Bitmap _bitmap;

        public Kernel Kernel
        {
            get;
        }
        public ImageView ImageView
        {
            get;
        }

        public Size DisplaySize
        {
            get; private set;
        }

        public UiKernel(Kernel kernel)
        {
            Kernel = kernel;
            ImageView = new ImageView();
            ImageView.SizeChanged += SizeChanged;
            ImageView.Shown += SizeChanged;
            _bitmap = new Bitmap(10, 10, PixelFormat.Format32bppRgb);
            ImageView.Image = _bitmap;
        }

        private void SizeChanged(object sender, EventArgs e)
        {
            var size = ImageView.Size;
            if (size.Width > 0 && size.Height > 0)
            {
                DisplaySize = size;
            }
        }

        public void Display(ImageData data)
        {
            if (_bitmap == null || _bitmap.Width != data.Width || _bitmap.Height != data.Height)
            {
                _bitmap?.Dispose();
                _bitmap = new Bitmap(data.Width, data.Height, PixelFormat.Format32bppRgb);
                ImageView.Image = _bitmap;
            }
            using (var pixels = _bitmap.Lock())
            {
                Marshal.Copy(data.Buffer, 0, pixels.Data, data.Width * data.Height);
            }
            data.Dispose();
            ImageView.Invalidate();
        }
    }
}
