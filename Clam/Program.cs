using System;
using System.IO;
using System.Linq;
using JetBrains.Annotations;

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

    static class Program
    {
        static void Main(string[] args)
        {
            Environment.CurrentDirectory = Path.Combine(Environment.CurrentDirectory, "Kernels");

            RenderWindow window = null;
            while (true)
            {
                switch (ConsoleHelper.Menu("Main Menu",
                    new[]
                    {
                        "Select kernel",
                        "Edit global parameters",
                        "Edit kernel parameters",
                        "Run kernel",
                        "Quit"
                    }))
                {
                    case 0:
                        if (window != null)
                            window.Dispose();
                        window = SelectKernelWindow();
                        break;
                    case 1:
                        StaticSettings.Edit();
                        break;
                    case 2:
                        if (window == null)
                        {
                            Console.WriteLine("No kernel selected");
                            Console.ReadKey(true);
                        }
                        else
                            window.Renderer.Kernel.Configure();
                        break;
                    case 3:
                        if (window == null)
                        {
                            Console.WriteLine("No kernel selected");
                            Console.ReadKey(true);
                        }
                        else
                            window.Run();
                        break;
                    case 4:
                        return;
                }
            }
        }

        [CanBeNull]
        static string SelectKernel()
        {
            var xmlFiles = Directory.GetFiles(Environment.CurrentDirectory, "*.xml");
            if (xmlFiles.Length == 0)
            {
                Console.WriteLine("Error: No kernels found");
                Console.ReadKey(true);
                return null;
            }
            if (xmlFiles.Length == 1)
                return xmlFiles[0];
            return xmlFiles[ConsoleHelper.Menu("Select render kernel", xmlFiles.Select(Path.GetFileName).ToArray())];
        }

        [CanBeNull]
        static RenderWindow SelectKernelWindow()
        {
            var kernel = SelectKernel();
            if (kernel == null)
                return null;
            var window = new RenderWindow();
            var package = RenderPackage.LoadFromXml(window, kernel);
            if (package == null)
            {
                window.Dispose();
                return null;
            }
            window.Renderer = package.Value;
            return window;
        }

        static void Configure(this RenderKernel kernel)
        {
            while (true)
            {
                var options = kernel.Options.ToArray();
                var index = ConsoleHelper.Menu("Edit parameter", options.Select(
                    kvp => kvp.Key + (string.IsNullOrWhiteSpace(kvp.Value) ? "" : " = " + kvp.Value)).Concat(new[] { "Done" }).ToArray());
                if (index == options.Length)
                    break;
                var option = options[index];
                var value = ConsoleHelper.PromptValue(option.Key, option.Value);
                kernel.SetOption(option.Key, value);
            }
            Console.WriteLine("Recompiling...");
            kernel.Recompile();
            Console.WriteLine("Done");
        }
    }

    static class ConsoleHelper
    {
        public static T PromptParseValue<T>(string variableName, T oldValue, PromptParseDelegate<T> parser)
        {
            return PromptParse(string.Format("Enter value for {0} (old value {1})", variableName, oldValue), parser);
        }

        public delegate bool PromptParseDelegate<T>(string str, out T value);
        public static T PromptParse<T>(string header, PromptParseDelegate<T> parser)
        {
            while (true)
            {
                var prompt = Prompt(header);
                T value;
                if (parser(prompt, out value))
                    return value;
                Console.WriteLine("Could not parse parameter, try again");
                Console.ReadKey(true);
            }
        }

        public static string PromptValue(string variableName, string oldValue)
        {
            return Prompt(string.IsNullOrWhiteSpace(oldValue)
                    ? "Enter value for " + variableName
                    : string.Format("Enter value for {0} (old value {1})", variableName, oldValue));
        }

        public static string Prompt(string header)
        {
            Console.Clear();
            Console.WriteLine(header);
            Console.Write("> ");
            var line = Console.ReadLine();
            Console.Clear();
            return line;
        }

        public static int Menu(string header, string[] options)
        {
            if (options.Length == 0)
                return -1;
            var selected = 0;
            var oldBackground = Console.BackgroundColor;
            var oldForeground = Console.ForegroundColor;
            while (true)
            {
                Console.Clear();
                Console.WriteLine(header);
                for (var i = 0; i < options.Length; i++)
                {
                    if (i == selected)
                    {
                        Console.BackgroundColor = oldForeground;
                        Console.ForegroundColor = oldBackground;
                        Console.WriteLine("{0}) {1}", i + 1, options[i]);
                        Console.BackgroundColor = oldBackground;
                        Console.ForegroundColor = oldForeground;
                    }
                    else
                        Console.WriteLine("{0}) {1}", i + 1, options[i]);
                }
                Console.SetCursorPosition(0, selected + 1);
                var winHeight = Console.WindowHeight;
                var position = selected - winHeight / 2;
                if (position > options.Length - winHeight)
                    position = options.Length - winHeight + 1;
                if (position < 0)
                    position = 0;
                Console.SetWindowPosition(0, position);

                var key = Console.ReadKey(true).Key;
                if (key == ConsoleKey.UpArrow)
                    selected--;
                else if (key == ConsoleKey.DownArrow)
                    selected++;
                else if (key == ConsoleKey.Enter)
                    break;
                else
                    Console.Beep();
                selected = Math.Max(Math.Min(selected, options.Length - 1), 0);
            }
            Console.Clear();
            return selected;
        }
    }
}
