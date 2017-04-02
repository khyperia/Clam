using System;
using System.Diagnostics;

namespace Clam4
{
    internal class TimeEstimator
    {
        private readonly Stopwatch _stopwatch;

        public TimeEstimator()
        {
            _stopwatch = Stopwatch.StartNew();
        }

        public TimeSpan Estimate(double currentProgress)
        {
            if (currentProgress == 0)
            {
                return TimeSpan.Zero;
            }
            var elapsed = _stopwatch.Elapsed;
            // rate = (currentProgress / elapsed)
            // rate * total = 1
            // total = 1 / rate
            // total = elapsed / currentProgress
            // estimated = total - elapsed
            // estimated = elapsed / currentProgress - elapsed
            var total = TimeSpan.FromTicks((long)(elapsed.Ticks / currentProgress));
            return total - elapsed;
        }
    }
}