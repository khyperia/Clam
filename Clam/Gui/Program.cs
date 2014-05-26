using System;
using System.IO;
using System.Windows.Forms;

namespace Clam.Gui
{
    static class Program
    {
        [STAThread]
        public static void Main()
        {
            var newPath = Path.Combine(Environment.CurrentDirectory, "Kernels");
            if (Directory.Exists(newPath))
                Environment.CurrentDirectory = newPath;
            else
            {
                MessageBox.Show("Error", "Kernels directory did not exist");
                return;
            }
            Application.Run(new MainWindow());
        }
    }
}