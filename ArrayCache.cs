using System;
using System.Collections.Concurrent;

namespace Clam4
{
    internal class ArrayCache<T>
    {
        private readonly int _maxSize;
        ConcurrentQueue<T[]> _queue;

        public ArrayCache(int maxSize)
        {
            _maxSize = maxSize;
            _queue = new ConcurrentQueue<T[]>();
        }

        public void Free(T[] array)
        {
            // ignore race condition with size
            if (_queue.Count < _maxSize)
            {
                _queue.Enqueue(array);
            }
        }

        public T[] Alloc(int size)
        {
            var dropped = false;
            T[] result;
            while (true)
            {
                if (_queue.TryDequeue(out result))
                {
                    if (result.Length == size)
                    {
                        break;
                    }
                    else
                    {
                        // drop array
                        dropped = true;
                        continue;
                    }
                }
                else
                {
                    result = new T[size];
                    break;
                }
            }
            if (dropped)
            {
                // without this, the heap slowly grows over time to GB amounts when resizing windows/etc.
                // with it, it keeps it at a proper (<100MB) size.
                GC.Collect();
            }
            return result;
        }
    }
}
