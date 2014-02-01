using System;

namespace Clam
{
    static class StaticSettings
    {
        private static int _screenshotHeight = 2048;
        public static int ScreenshotHeight { get { return _screenshotHeight; } }
        private static int _screenshotPartialRender = 10;
        public static int ScreenshotPartialRender { get { return _screenshotPartialRender; } }

        public static void Edit()
        {
            while (true)
            {
                switch (ConsoleHelper.Menu("Select option", new[]
                {
                    "Screenshot height",
                    "Screenshot partial render",
                    "What is screenshot partial render? (help text)",
                    "Done"
                }))
                {
                    case 0:
                        _screenshotHeight = ConsoleHelper.PromptParseValue("screenshot height", _screenshotHeight, int.TryParse);
                        break;
                    case 1:
                        _screenshotPartialRender = ConsoleHelper.PromptParseValue("screenshot partial render", _screenshotPartialRender, int.TryParse);
                        break;
                    case 2:
                        Console.WriteLine("Screenshot partial render help:");
                        Console.WriteLine("The render engine, by default, subdivides the screen into N^2 rectangles");
                        Console.WriteLine("where N is the parameter controlled by \"screenshot partial render\"");
                        Console.WriteLine("Each rectangle is rendered in a seperate kernel call");
                        Console.WriteLine("This cuts down on time spent in any one kernel call");
                        Console.WriteLine("If the screen flashes, GPU hangs, and Clam crashes,");
                        Console.WriteLine("this is Windows hitting it's two-second limit on kernel calls");
                        Console.WriteLine("Increase this parameter in such a case");
                        Console.ReadKey(true);
                        break;
                    case 3:
                        return;
                }
            }
        }
    }
}