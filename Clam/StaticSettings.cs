namespace Clam
{
    public class StaticSettings
    {
        public static readonly StaticSettings Fetch = new StaticSettings();

        private StaticSettings()
        {
            OpenClOptions = "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror";
            GifFramerate = 20;
            GifFramecount = 20;
            GifHeight = 256;
            ScreenshotPartialRender = 10;
            ScreenshotHeight = 2048;
        }

        public int ScreenshotHeight;
        public int ScreenshotPartialRender;
        public int GifHeight;
        public int GifFramecount;
        public int GifFramerate;
        public string OpenClOptions;
    }
}