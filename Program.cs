using Cloo;
using Eto.Forms;
using System;

namespace Clam4
{
    internal class Program
    {
        public static readonly string HelpText = @"Buttons:
Start/Stop: this starts and stops the rendering.
Save/Load State: Save/load the giant list of settings and their values to a file, to be re-loaded later. (E.g. to save a camera position)
Run headless render: Run a long-running, high-quality render, and save the result to a file.
Bunches of text/settings: Individual parameters for the fractal. I would advise against manually editing ""look""/""up"" X/Y/Z, as weird things happen.

Press Escape or click the fractal to have key presses go to controlling the fractal.

Keybindings:
Esc: pressing escape at any point when editing text on the left will bring you back to the fractal.
WASD and space/Z: Move the camera forwards, left, back, right, up, down.
IJKL or arrow keys: Pan the camera up, down, left, right
UO or QE: Roll the camera left and right
RF: Adjust the speed the camera moves at, as well as the focal plane (focus) for depth-of-field.
NM: Adjust the field-of-view (zoom).";

        [STAThread]
        public static void Main(string[] args)
        {
            Kernel kernel;
            try
            {
                //kernel = new Mandelbrot(device);
                kernel = new Mandelbox(Kernel.GetContext());
            }
            catch (Kernel.CompilationException)
            {
                return;
            }
            var app = new Application();
            var kernelControl = new UiKernel(kernel);
            var settings = kernel.Defaults;
            var keyTracker = new KeyTracker(kernel.KeyHandlers, settings);
            var model = new UiModel(kernelControl, keyTracker);
            var form = UiView.Create(model);
            app.Run(form);
            Console.WriteLine("Exiting");
        }
    }
}