using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using System.Xml.Linq;
using Cloo;
using JetBrains.Annotations;

namespace Clam.Gui
{
    class MainWindow : Form
    {
        private ListBox gpuListBox;
        private ListBox kernelListBox;
        private Button saveButton;
        private Button loadButton;
        private Button screenshotButton;
        private Button applyButton;
        [CanBeNull]
        private RenderWindow renderWindow;

        public MainWindow()
        {
            InitializeComponent();
            RefreshGpus();
            RefreshXmlFiles();
        }

        private void RefreshGpus()
        {
            gpuListBox.Items.Clear();
            foreach (var device in ComputePlatform.Platforms.SelectMany(p => p.Devices))
                gpuListBox.Items.Add(device);
        }

        private void RefreshXmlFiles()
        {
            kernelListBox.Items.Clear();
            foreach (var xmlFile in Directory.EnumerateFiles(Environment.CurrentDirectory, "*.xml")
                .SelectMany(xmlFile => KernelXmlFile.Load(XElement.Load(xmlFile))))
                kernelListBox.Items.Add(xmlFile);
        }

        private void RefreshDefines()
        {
            if (renderWindow == null)
                return;
            for (var i = 0; i < Controls.Count; i++)
            {
                if (Controls[i].Name.StartsWith("temp_") == false)
                    continue;
                Controls.RemoveAt(i);
                i--;
            }

            if (renderWindow.Renderer.Kernel == null)
                return;

            var y = 320;
            AddConfigOption(ref y, "OpenClOptions", StaticSettings.Fetch.OpenClOptions,
                s => StaticSettings.Fetch.OpenClOptions = s);
            AddConfigOption(ref y, "ScreenshotHeight", StaticSettings.Fetch.ScreenshotHeight.ToString(CultureInfo.InvariantCulture),
                s => int.TryParse(s, out StaticSettings.Fetch.ScreenshotHeight));
            AddConfigOption(ref y, "ScreenshotPartialRender", StaticSettings.Fetch.ScreenshotPartialRender.ToString(CultureInfo.InvariantCulture),
                s => int.TryParse(s, out StaticSettings.Fetch.ScreenshotPartialRender));
            AddConfigOption(ref y, "GifFramecount", StaticSettings.Fetch.GifFramecount.ToString(CultureInfo.InvariantCulture),
                s => int.TryParse(s, out StaticSettings.Fetch.GifFramecount));
            AddConfigOption(ref y, "GifFramerate", StaticSettings.Fetch.GifFramerate.ToString(CultureInfo.InvariantCulture),
                s => int.TryParse(s, out StaticSettings.Fetch.GifFramerate));
            AddConfigOption(ref y, "GifHeight", StaticSettings.Fetch.GifHeight.ToString(CultureInfo.InvariantCulture),
                s => int.TryParse(s, out StaticSettings.Fetch.GifHeight));

            foreach (var kvp in renderWindow.Renderer.Kernel.Options)
            {
                var kvp1 = kvp;
                AddConfigOption(ref y, kvp.Key, kvp.Value, s =>
                {
                    if (renderWindow.Renderer.Kernel == null)
                        return;
                    renderWindow.Renderer.Kernel.SetOption(kvp1.Key, s);
                });
            }
        }

        private void AddConfigOption(ref int y, string kvpKey, string kvpValue, Action<string> set)
        {
            if (renderWindow == null)
                return;

            var name = new TextBox();
            name.Location = new Point(0, y);
            name.Size = new Size(100, 23);
            name.Name = "temp_n" + y;
            name.Text = kvpKey;
            name.Enabled = false;

            var value = new TextBox();
            value.Location = new Point(100, y);
            value.Size = new Size(100, 23);
            value.Name = "temp_v" + y;
            value.Text = kvpValue;
            value.Enabled = true;
            value.TextChanged += (o, e) => set(value.Text);

            Controls.Add(name);
            Controls.Add(value);

            y += 20;
        }

        private void InitializeComponent()
        {
            this.kernelListBox = new System.Windows.Forms.ListBox();
            this.gpuListBox = new System.Windows.Forms.ListBox();
            this.saveButton = new System.Windows.Forms.Button();
            this.loadButton = new System.Windows.Forms.Button();
            this.screenshotButton = new System.Windows.Forms.Button();
            this.applyButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // kernelListBox
            // 
            this.kernelListBox.FormattingEnabled = true;
            this.kernelListBox.Location = new System.Drawing.Point(0, 62);
            this.kernelListBox.Name = "kernelListBox";
            this.kernelListBox.Size = new System.Drawing.Size(200, 199);
            this.kernelListBox.TabIndex = 0;
            this.kernelListBox.SelectedIndexChanged += new System.EventHandler(this.kernelListBox_SelectedIndexChanged);
            // 
            // gpuListBox
            // 
            this.gpuListBox.FormattingEnabled = true;
            this.gpuListBox.Location = new System.Drawing.Point(0, 0);
            this.gpuListBox.Name = "gpuListBox";
            this.gpuListBox.Size = new System.Drawing.Size(200, 56);
            this.gpuListBox.TabIndex = 1;
            this.gpuListBox.SelectedIndexChanged += new System.EventHandler(this.gpuListBox_SelectedIndexChanged);
            // 
            // saveButton
            // 
            this.saveButton.Location = new System.Drawing.Point(0, 262);
            this.saveButton.Name = "saveButton";
            this.saveButton.Size = new System.Drawing.Size(100, 23);
            this.saveButton.TabIndex = 2;
            this.saveButton.Text = "Save";
            this.saveButton.UseVisualStyleBackColor = true;
            this.saveButton.Click += new System.EventHandler(this.saveButton_Click);
            // 
            // loadButton
            // 
            this.loadButton.Location = new System.Drawing.Point(100, 262);
            this.loadButton.Name = "loadButton";
            this.loadButton.Size = new System.Drawing.Size(100, 23);
            this.loadButton.TabIndex = 3;
            this.loadButton.Text = "Load";
            this.loadButton.UseVisualStyleBackColor = true;
            this.loadButton.Click += new System.EventHandler(this.loadButton_Click);
            // 
            // screenshotButton
            // 
            this.screenshotButton.Location = new System.Drawing.Point(0, 287);
            this.screenshotButton.Name = "screenshotButton";
            this.screenshotButton.Size = new System.Drawing.Size(100, 23);
            this.screenshotButton.TabIndex = 4;
            this.screenshotButton.Text = "Screenshot";
            this.screenshotButton.UseVisualStyleBackColor = true;
            this.screenshotButton.Click += new System.EventHandler(this.screenshotButton_Click);
            // 
            // applyButton
            // 
            this.applyButton.Location = new System.Drawing.Point(100, 287);
            this.applyButton.Name = "applyButton";
            this.applyButton.Size = new System.Drawing.Size(100, 23);
            this.applyButton.TabIndex = 6;
            this.applyButton.Text = "Apply";
            this.applyButton.UseVisualStyleBackColor = true;
            this.applyButton.Click += new System.EventHandler(this.applyButton_Click);
            // 
            // MainWindow
            // 
            this.ClientSize = new System.Drawing.Size(1000, 600);
            this.Controls.Add(this.applyButton);
            this.Controls.Add(this.screenshotButton);
            this.Controls.Add(this.loadButton);
            this.Controls.Add(this.saveButton);
            this.Controls.Add(this.gpuListBox);
            this.Controls.Add(this.kernelListBox);
            this.Name = "MainWindow";
            this.ResumeLayout(false);

        }

        private void gpuListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (renderWindow != null)
            {
                Controls.Remove(renderWindow);
                renderWindow.Dispose();
            }
            renderWindow = new RenderWindow((ComputeDevice)gpuListBox.SelectedItem, s => Text = s);
            renderWindow.Location = new Point(200, 0);
            renderWindow.Size = new Size(800, 600);
            renderWindow.Name = "renderWindow";
            Controls.Add(renderWindow);
        }

        private void kernelListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (renderWindow == null)
                return;
            var kernel = (KernelXmlFile)kernelListBox.SelectedItem;
            var package = RenderPackage.LoadFromXml(renderWindow.ComputeContext, kernel);
            if (package.HasValue == false)
                return;
            renderWindow.Renderer = package.Value;

            RefreshDefines();
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            var selector = new SaveFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                DefaultExt = ".kernel.xml",
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == DialogResult.OK)
                renderWindow.SaveProgram((KernelXmlFile)kernelListBox.SelectedItem, selector.FileName);
        }

        private void loadButton_Click(object sender, EventArgs e)
        {
            var selector = new OpenFileDialog
            {
                InitialDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "State")),
                Filter = "Xml kernel files|*.kernel.xml"
            };
            if (selector.ShowDialog(this) == DialogResult.OK)
                renderWindow.LoadProgram(selector.FileName);
        }

        private void screenshotButton_Click(object sender, EventArgs e)
        {
            if (renderWindow == null)
                return;
            ThreadPool.QueueUserWorkItem(o =>
            {
                var screenshot = renderWindow.Renderer.Screenshot(StaticSettings.Fetch.ScreenshotHeight,
                    StaticSettings.Fetch.ScreenshotPartialRender, renderWindow.DisplayInformation);
                renderWindow.DisplayInformation("Saving screenshot");
                if (screenshot != null)
                    screenshot.Save(Ext.UniqueFilename("screenshot", "png"));
                renderWindow.DisplayInformation("Done taking screenshot");
            });
        }

        private void applyButton_Click(object sender, EventArgs e)
        {
            if (renderWindow == null || renderWindow.Renderer.Kernel == null)
                return;
            renderWindow.Renderer.Kernel.Recompile();
            var framedepcontrols = renderWindow.Renderer.Parameters as IFrameDependantControl;
            if (framedepcontrols != null)
                framedepcontrols.Frame = 0;
        }
    }
}
