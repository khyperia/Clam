using System;
using System.IO;
using System.Xml.Linq;
using OpenTK;

namespace Clam
{
    static class Ext
    {
        public static string UniqueFilename(string filename, string ext)
        {
            return UniqueFileInDirectory(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), filename, ext);
        }

        public static string UniqueFileInDirectory(string directory, string filename, string ext)
        {
            var screenshotNumber = 0;
            string file;
            while (File.Exists(file = Path.Combine(directory, string.Format("{0}{1}.{2}", filename, screenshotNumber, ext))))
                screenshotNumber++;
            return file;
        }

        //public static string UniqueDirectory(string dirname)
        //{
        //    var screenshotNumber = 0;
        //    string file;
        //    while (Directory.Exists(file = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), string.Format("{0}{1}", dirname, screenshotNumber))))
        //        screenshotNumber++;
        //    return file;
        //}

        public static XElement Save(this Vector3d vector, string name)
        {
            return new XElement(name,
                new XElement("X", vector.X),
                new XElement("Y", vector.Y),
                new XElement("Z", vector.Z));
        }

        public static Vector3d LoadVector3D(this XElement element)
        {
            return new Vector3d(
                element.Element("X").LoadDouble(),
                element.Element("Y").LoadDouble(),
                element.Element("Z").LoadDouble()
                );
        }

        public static XElement Save(this float value, string name)
        {
            return new XElement(name, value);
        }

        public static float LoadFloat(this XElement element)
        {
            return float.Parse(element.Value);
        }

        public static XElement Save(this double value, string name)
        {
            return new XElement(name, value);
        }

        public static double LoadDouble(this XElement element)
        {
            return double.Parse(element.Value);
        }
    }
}
