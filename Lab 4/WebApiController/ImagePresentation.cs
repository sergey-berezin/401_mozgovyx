using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.CompilerServices;

namespace WebApiController
{
    public class ImagePresentation
    {
        public byte[] Image { get; set; }
        public string ClassName { get; set; }
        public double Confidence { get; set; }

        public ImagePresentation(Image<Rgb24> image, string className, double confidence)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                image.Save(ms, JpegFormat.Instance);
                Image = ms.ToArray();
            }
            ClassName = className;
            Confidence = confidence;
        }
    }
}
