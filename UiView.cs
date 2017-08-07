using Eto.Drawing;
using Eto.Forms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Clam4
{
    internal static class UiView
    {
        public static Form Create(UiModel model)
        {
            return new Form()
            {
                Title = "Clam4",
                Content = MainView(model),
                Width = 1000,
                Height = 600,
            };
        }

        private static Control MainView(UiModel model)
        {
            var panel = new List<Control>();

            var help = new Button((o, e) => MessageBox.Show(Program.HelpText, "Help"))
            {
                Text = "Help (keybindings, etc)",
            };
            panel.Add(help);

            Button startStop = null;
            startStop = new Button((o, e) =>
            {
                if (model.StartStop())
                {
                    startStop.Text = "Stop";
                }
                else
                {
                    startStop.Text = "Start";
                }
            })
            {
                Text = "Start",
            };
            panel.Add(startStop);

            var settingsFilename = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Clam4", "settings.clam4");
            var settingsDirectory = Path.GetDirectoryName(settingsFilename);
            var saveState = new Button((o, e) =>
            {
                if (!Directory.Exists(settingsDirectory))
                {
                    Directory.CreateDirectory(settingsDirectory);
                }
                model.Settings.Save(settingsFilename);
            })
            {
                Text = "Save state",
            };
            panel.Add(saveState);

            var loadState = new Button((o, e) => model.Settings.Load(settingsFilename))
            {
                Text = "Load state",
            };
            panel.Add(loadState);

            var focusThing = new Button()
            {
                Text = "Control camera (focus)",
            };
            panel.Add(focusThing);

            foreach (var setting in model.Settings)
            {
                var key = setting.Key;

                var label = new Label()
                {
                    TextAlignment = TextAlignment.Right,
                    Text = key,
                };

                var text = new TextBox()
                {
                    Text = setting.Value.ToString(),
                };
                var changing = false;
                model.Settings.OnChange += (changedKey, changedValue) =>
                {
                    if (changing || text.HasFocus)
                    {
                        return;
                    }
                    if (changedKey == key)
                    {
                        changing = true;
                        text.Text = changedValue.ToString();
                        changing = false;
                    }
                };
                text.TextChanged += (o, e) =>
                {
                    if (changing)
                    {
                        return;
                    }
                    model.SetSetting(key, text.Text);
                };
                panel.Add(HorizontalExpandAll(label, text));
            }

            Headless(panel, model);

            var wholePanel = Vertical(panel.ToArray());
            wholePanel.MinimumSize = new Size(300, -1);
            var scroll = new Scrollable { Content = wholePanel };
            var wholeWindow = HorizontalExpandLast(scroll, model.UiKernel.ImageView);
            Keybinds(wholeWindow, focusThing, model);
            return wholeWindow;
        }

        private static void Headless(List<Control> panel, UiModel model)
        {
            panel.Add(new Label { Text = "Headless render" });
            var filenameText = new TextBox() { Text = "render.png" };
            panel.Add(HorizontalExpandLast(new Label { Text = "Save as" }, filenameText));
            var widthText = new TextBox() { Text = "1000" };
            panel.Add(HorizontalExpandLast(new Label { Text = "Width" }, widthText));
            var heightText = new TextBox() { Text = "1000" };
            panel.Add(HorizontalExpandLast(new Label { Text = "Height" }, heightText));
            var framesText = new TextBox() { Text = "100" };
            panel.Add(HorizontalExpandLast(new Label { Text = "Frames" }, framesText));
            Button goButton = null;

            panel.Add(goButton = new Button((o, e) =>
            {
                var filename = filenameText.Text;
                if (!Path.IsPathRooted(filename))
                {
                    filename = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Clam4", filename);
                }
                if (int.TryParse(widthText.Text, out var width) &&
                    int.TryParse(heightText.Text, out var height) &&
                    int.TryParse(framesText.Text, out var frames))
                {
                    goButton.Enabled = false;
                    model.RunHeadless(width, height, frames, filename, goButton);
                }
            })
            {
                Text = "Go!"
            });
        }

        private static void Keybinds(Control window, Control focusThing, UiModel model)
        {
            //model.UiKernel.Click += (o, e) => model.UiKernel.Select();
            window.KeyDown += (o, e) =>
            {
                if (e.Key == Keys.Escape)
                {
                    e.Handled = true;
                    focusThing.Focus();
                    return;
                }
            };
            // model.UiKernel.ImageView.PreviewKeyDown += (o, e) =>
            // {
            //     var code = e.KeyCode;
            //     var isArrow = code == Keys.Up || code == Keys.Down || code == Keys.Left || code == Keys.Right;
            //     if (isArrow)
            //     {
            //         e.IsInputKey = true;
            //     }
            // };
            focusThing.KeyDown += (o, e) =>
            {
                if (model.OnKeyDown(e.Key))
                {
                    e.Handled = true;
                }
            };
            focusThing.KeyUp += (o, e) =>
            {
                if (model.OnKeyUp(e.Key))
                {
                    e.Handled = true;
                }
            };
        }

        private static StackLayout Vertical(params Control[] controls)
            => Stack(false, controls.Select(c => new StackLayoutItem(c)));
        private static StackLayout VerticalExpandAll(params Control[] controls)
            => Stack(false, controls.Select(c => new StackLayoutItem(c, true)));
        private static StackLayout VerticalExpandFirst(params Control[] controls)
            => Stack(false, controls.Select((c, i) => new StackLayoutItem(c, i == 0)));
        private static StackLayout VerticalExpandLast(params Control[] controls)
            => Stack(false, controls.Select((c, i) => new StackLayoutItem(c, i == controls.Length - 1)));
        private static StackLayout Horizontal(params Control[] controls)
            => Stack(true, controls.Select(c => new StackLayoutItem(c)));
        private static StackLayout HorizontalExpandAll(params Control[] controls)
            => Stack(true, controls.Select(c => new StackLayoutItem(c, true)));
        private static StackLayout HorizontalExpandFirst(params Control[] controls)
            => Stack(true, controls.Select((c, i) => new StackLayoutItem(c, i == 0)));
        private static StackLayout HorizontalExpandLast(params Control[] controls)
            => Stack(true, controls.Select((c, i) => new StackLayoutItem(c, i == controls.Length - 1)));
        private static StackLayout Stack(bool horizontal, IEnumerable<StackLayoutItem> controls)
        {
            var stack = new StackLayout();
            if (horizontal)
            {
                stack.Orientation = Orientation.Horizontal;
                stack.VerticalContentAlignment = VerticalAlignment.Stretch;
            }
            else
            {
                stack.Orientation = Orientation.Vertical;
                stack.HorizontalContentAlignment = HorizontalAlignment.Stretch;
            };
            foreach (var control in controls)
            {
                stack.Items.Add(control);
            }
            return stack;
        }
    }
}