using System.Linq;
using System.Windows.Forms;
using System.Xml.Linq;

namespace Clam
{
    public static class WholeProgramSerializer
    {
        public static void SaveProgram(this RenderWindow renderWindow, KernelXmlFile kernel, string filename)
        {
            if (renderWindow.Renderer.Kernel == null)
                return;
            var kernelOptions = renderWindow.Renderer.Kernel.SerializeOptions("KernelOptions");
            var serializableParams = renderWindow.Renderer.Parameters as ISerializableParameterSet;
            var controlOptions = serializableParams == null ? null : serializableParams.Save("ControlOptions");
            var element = new XElement("ClamConfig", kernel.Save("Kernel"), kernelOptions);
            if (controlOptions != null)
                element.Add(controlOptions);
            element.Save(filename);
        }

        public static void LoadProgram(this RenderWindow renderWindow, string filename)
        {
            var doc = XElement.Load(filename);
            if (doc.Name.LocalName != "ClamConfig")
            {
                MessageBox.Show("XML was not a ClamConfig file", "Error");
                return;
            }
            var kernelElement = doc.Element("Kernel");
            var kernelOptionsElement = doc.Element("KernelOptions");
            if (kernelElement == null || kernelOptionsElement == null)
            {
                MessageBox.Show("XML settings are incomplete", "Error");
                return;
            }
            var kxf = KernelXmlFile.Load(kernelElement).Single();

            var package = RenderPackage.LoadFromXml(renderWindow.ComputeContext, kxf, renderWindow.Renderer.Parameters);
            if (package.HasValue == false)
                return;

            renderWindow.Renderer = package.Value;

            if (renderWindow.Renderer.Kernel == null)
                return;

            renderWindow.Renderer.Kernel.LoadOptions(kernelOptionsElement);

            var serializable = renderWindow.Renderer.Parameters as ISerializableParameterSet;
            if (serializable != null)
            {
                var element = doc.Element("ControlOptions");
                if (element != null)
                    serializable.Load(element);
            }
        }
    }
}