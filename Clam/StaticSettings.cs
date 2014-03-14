using System.ComponentModel;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace Clam
{
    public class StaticSettings : INotifyPropertyChanged
    {
        public static StaticSettings Fetch = new StaticSettings();

        private int _screenshotHeight = 2048;
        private int _screenshotPartialRender = 10;
        private int _gifHeight = 256;
        private int _gifFramecount = 20;
        private int _gifFramerate = 20;

        public int ScreenshotHeight
        {
            get { return _screenshotHeight; }
            set
            {
                if (value == _screenshotHeight) return;
                _screenshotHeight = value;
                OnPropertyChanged();
            }
        }

        public int ScreenshotPartialRender
        {
            get { return _screenshotPartialRender; }
            set
            {
                if (value == _screenshotPartialRender) return;
                _screenshotPartialRender = value;
                OnPropertyChanged();
            }
        }

        public int GifHeight
        {
            get { return _gifHeight; }
            set
            {
                if (value == _gifHeight) return;
                _gifHeight = value;
                OnPropertyChanged();
            }
        }

        public int GifFramecount
        {
            get { return _gifFramecount; }
            set
            {
                if (value == _gifFramecount) return;
                _gifFramecount = value;
                OnPropertyChanged();
            }
        }

        public int GifFramerate
        {
            get { return _gifFramerate; }
            set
            {
                if (value == _gifFramerate) return;
                _gifFramerate = value;
                OnPropertyChanged();
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            var handler = PropertyChanged;
            if (handler != null) handler(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}