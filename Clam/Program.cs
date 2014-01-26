using System;
using System.IO;
using System.Linq;
using JetBrains.Annotations;

namespace Clam
{
    class Program
    {
        static void Main(string[] args)
        {
            Environment.CurrentDirectory = Path.Combine(Environment.CurrentDirectory, "Kernels");

            var kernel = SelectKernel();
            if (kernel == null)
                Console.WriteLine("Error: No kernels found");
            else
            {
                Console.WriteLine("Using kernel " + kernel);
                RunKernel(kernel);
            }
        }

        [CanBeNull]
        static string SelectKernel()
        {
            var xmlFiles = Directory.GetFiles(Environment.CurrentDirectory, "*.xml");
            if (xmlFiles.Length == 0)
                return null;
            if (xmlFiles.Length == 1)
                return xmlFiles[0];
            return xmlFiles[ConsoleHelper.Menu("Select render kernel", xmlFiles.Select(s => Path.GetFileName(s)).ToArray())];
        }

        static void RunKernel([NotNull] string kernelFile)
        {
            var window = new RenderWindow();
            var kernel = RenderPackage.LoadFromXml(window, kernelFile);
            if (kernel == null)
                Console.ReadKey(true);
            else
            {
                window.Renderer = kernel.Value;
                window.Run();
            }
            window.Dispose();
        }
    }

    static class ConsoleHelper
    {
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
