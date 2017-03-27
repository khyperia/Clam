using ManagedCuda;
using System.Threading.Tasks;

namespace Clam4
{
    internal class Program
    {
        public static readonly string HelpText = @"Buttons:
start/stop: this starts and stops the rendering.
Focus keybindings: when this button is clicked or highlighted, key presess will be registered to controlling the fractal.
Bunches of text/settings: Individual parameters for the fractal. I would avise against manually editing look/up X/Y/Z, as weird things happen.

Keybindings:
Esc: pressing esacpe at any point when editing text on the left will highlight the ""Focus Keybindings"" button and bring you back to the fractal.
WASD and space/Z: Move the camera forwards, left, back, right, up, down.
IJKL: Pan the camera up, down, left, right
UO or QE: Roll the camera left and right
RF: Adjust the speed the camera moves at, as well as the focal plane (focus) for depth-of-field.
NM: Adjust the field-of-view (zoom).";

        public static void Main(string[] args)
        {
            var device = new CudaContext();
            Kernel render;
            try
            {
                //render = new Mandelbrot(device);
                render = new Mandelbox(device);
            }
            catch (Kernel.CompilationException)
            {
                return;
            }
            var window = new GuiWindow(render, UiControls.UiWidth);
            var settings = render.Defaults;
            var keyhandler = new KeyTracker(render.KeyHandlers, settings);
            UiControls.Create(window.Form, settings, keyhandler.OnKeyDown, keyhandler.OnKeyUp, async cancellation =>
            {
                const int concurrentTasks = 1;
                var tasks = new Task<ImageData>[concurrentTasks];
                while (!cancellation.IsCancellationRequested)
                {
                    for (var i = 0; i < tasks.Length; i++)
                    {
                        if (tasks[i] != null)
                        {
                            var image = await tasks[i];
                            window.Display(image);
                        }
                        keyhandler.OnRender();
                        tasks[i] = render.Run(settings);
                    }
                }
            });
            window.Run();
        }
    }
}