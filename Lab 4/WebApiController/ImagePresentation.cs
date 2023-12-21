using SixLabors.ImageSharp;
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
            Image = new byte[image.Width * image.Height * Unsafe.SizeOf<Rgb24>()];
            image.CopyPixelDataTo(Image);
            ClassName = className;
            Confidence = confidence;
        }
    }
}
