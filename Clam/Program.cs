using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml.Linq;
using JetBrains.Annotations;
using OpenTK;

namespace Clam
{
    static class Program
    {
        static void Main()
        {
            var newPath = Path.Combine(Environment.CurrentDirectory, "Kernels");
            if (Directory.Exists(newPath))
                Environment.CurrentDirectory = newPath;
            else
                Console.WriteLine("Kernels directory didn't exist");

            Console.WriteLine("Creating OpenGL context");
            var window = new RenderWindow { WindowState = WindowState.Minimized };
            new Thread(Menu).Start(window);
            window.Run();
            window.Dispose();
        }

        static void Menu(object windowObject)
        {
            var window = (RenderWindow)windowObject;
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
                        SelectKernelWindow(window);
                        break;
                    case 1:
                        StaticSettings.Edit();
                        break;
                    case 2:
                        ConfigureKernel(window);
                        break;
                    case 3:
                        window.WindowState = WindowState.Normal;
                        break;
                    case 4:
                        window.Invoke(() =>
                        {
                            window.CancelClosing = false;
                            window.Exit();
                        });
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

        static void SelectKernelWindow(RenderWindow window)
        {
            var kernel = SelectKernel();
            if (kernel == null)
                return;
            window.Invoke(() =>
            {
                var package = RenderPackage.LoadFromXml(window, kernel);
                if (package == null)
                    return;
                window.Renderer = package.Value;
            });
        }

        static void ConfigureKernel(RenderWindow window)
        {
            var kernel = window.Renderer.Kernel;
            if (kernel == null)
            {
                Console.WriteLine("No kernel selected");
                Console.ReadKey(true);
                return;
            }
            while (true)
            {
                var options = kernel.Options.ToArray();
                var index = ConsoleHelper.Menu("Edit parameter", options.Select(
                    kvp => kvp.Key + (string.IsNullOrWhiteSpace(kvp.Value) ? "" : " = " + kvp.Value)).Concat(new[] { "Save", "Load", "Done" }).ToArray());
                if (index == options.Length)
                {
                    SaveKernel(kernel);
                    continue;
                }
                if (index == options.Length + 1)
                {
                    LoadKernel(kernel);
                    continue;
                }
                if (index == options.Length + 2)
                    break;
                var option = options[index];
                var value = ConsoleHelper.PromptValue(option.Key, option.Value);
                kernel.SetOption(option.Key, value);
            }
            window.Invoke(kernel.Recompile);
        }

        static void SaveKernel(RenderKernel kernel)
        {
            var directory = Path.Combine("..", "State");
            if (Directory.Exists(directory) == false)
                Directory.CreateDirectory(directory);
            var list = Directory.GetFiles(directory, "*.kernel.xml");
            var index = ConsoleHelper.Menu("Save to where", list.Select(f => string.Format("Overwrite {0}", Path.GetFileName(f))).Concat(new[] { "New state", "Cancel" }).ToArray());
            string filename;
            if (index == list.Length)
            {
                var prompt = ConsoleHelper.Prompt("Enter filename (empty for autogen name)");
                if (string.IsNullOrWhiteSpace(prompt))
                    filename = Ext.UniqueFileInDirectory(directory, "state", "kernel.xml");
                else
                {
                    if (prompt.EndsWith(".kernel.xml") == false)
                        prompt += ".kernel.xml";
                    filename = Path.Combine(directory, prompt);
                }
            }
            else if (index == list.Length + 1)
                return;
            else
                filename = list[index];
            var xml = kernel.SerializeOptions();
            xml.Save(filename);
        }

        static void LoadKernel(RenderKernel kernel)
        {
            var directory = Path.Combine("..", "State");
            if (Directory.Exists(directory) == false)
            {
                Console.WriteLine("State directory does not exist");
                return;
            }
            var list = Directory.GetFiles(directory, "*.kernel.xml");
            if (list.Length == 0)
            {
                Console.WriteLine("No state files found");
                return;
            }
            var index = ConsoleHelper.Menu("Load kernel", list.Select(Path.GetFileName).Concat(new[] { "Cancel" }).ToArray());
            if (index == list.Length)
                return;
            kernel.LoadOptions(XElement.Load(list[index]));
        }
    }

    static class ConsoleHelper
    {
        public static void Alert(string message)
        {
            Console.Clear();
            Console.WriteLine(message);
        }

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
