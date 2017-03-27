using System;
using System.Threading;

namespace Clam4
{
    internal struct ImageData : IDisposable
    {
        private static ArrayCache<int> BufferCache = new ArrayCache<int>(16);

        public int Width { get; }
        public int Height { get; }
        public int[] Buffer { get => _buffer; }
        private int[] _buffer;

        public bool Valid => _buffer != null;

        public ImageData(int width, int height, int size)
            : this(width, height, BufferCache.Alloc(size))
        {
        }

        private ImageData(int width, int height, int[] buffer)
        {
            Width = width;
            Height = height;
            _buffer = buffer;
        }

        public bool TryDispose()
        {
            var old = Interlocked.Exchange(ref _buffer, null);
            if (old != null)
            {
                BufferCache.Free(old);
                return true;
            }
            return false;
        }

        public void Dispose()
        {
            TryDispose();
        }
    }
}