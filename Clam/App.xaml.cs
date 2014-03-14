using System;
using System.IO;
using System.Windows;
using Clam.Xaml;

namespace Clam
{
    public partial class App
    {
        public App()
        {
            var newPath = Path.Combine(Environment.CurrentDirectory, "Kernels");
            if (Directory.Exists(newPath))
                Environment.CurrentDirectory = newPath;
            else
            {
                MessageBox.Show("Error", "Kernels directory did not exist");
                return;
            }
            new MainWindow().Show();
        }
    }
}
