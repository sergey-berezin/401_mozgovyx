using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using YoloPackage;

namespace WebApiController
{
    public record SegmentedObject(Image<Rgb24> OriginalImage, string Class, double Confidence, Image<Rgb24> BoundingBox, ObjectBox bbox);

    public class ProcessingTools
    {
        private const int TargetSize = 416;
        private static readonly ResizeOptions resizeOptions = new ResizeOptions
        {
            Size = new Size(TargetSize, TargetSize),
            Mode = ResizeMode.Pad
        };
        public static async Task<IEnumerable<SegmentedObject>> FindImageSegmentation(string filename, CancellationToken token)
        {
            var originalImage = Image.Load<Rgb24>(filename);
            var processingTask = YoloService.ProcessImage(originalImage, token);
            List<SegmentedObject> imageSegmentation = new();

            await processingTask;
            var bboxes = processingTask.Result.BoundingBoxes;
            var imagesCut = await Task.WhenAll(
                bboxes.Select(bbox => Task.Run(
                    () => CutBoundingBox(originalImage, bbox), token
                )
            ));
            return imagesCut.Zip(bboxes).Select(
                arg => new SegmentedObject(
                    originalImage,
                    YoloService.Labels[arg.Second.Class],
                    arg.Second.Confidence,
                    arg.First,
                    arg.Second
                )
           );
        }

        public static Image<Rgb24> CutBoundingBox(Image<Rgb24> original, ObjectBox bbox)
        {
            int x = (int)bbox.XMin, y = (int)bbox.YMin;
            int width = (int)(bbox.XMax - bbox.XMin), height = (int)(bbox.YMax - bbox.YMin);
            if (x < 0)
            {
                width += x;
                x = 0;
            }
            if (y < 0)
            {
                height += y;
                y = 0;
            }
            if (x + width > TargetSize)
            {
                width = TargetSize - x;
            }
            if (y + height > TargetSize)
            {
                height = TargetSize - y;
            }
            if (x > TargetSize || y > TargetSize)
            {
                return original.Clone(
                    i => i.Resize(resizeOptions)
                );
            }

            return original.Clone(
                i => i.Resize(resizeOptions).Crop(new Rectangle(x, y, width, height))
            );
        }

        public static Image<Rgb24> Annotate(Image<Rgb24> target, ObjectBox bbox)
        {
            int maxDimension = Math.Max(target.Width, target.Height);
            float scale = (float)maxDimension / TargetSize;
            return target.Clone(ctx =>
            {
                ctx.Resize(new ResizeOptions { Size = new Size(maxDimension, maxDimension), Mode = ResizeMode.Pad}).DrawPolygon(
                    Pens.Solid(Color.Red, 1 + maxDimension / TargetSize),
                    new PointF[] {
                        new PointF((float)bbox.XMin * scale, (float) bbox.YMin * scale),
                        new PointF((float)bbox.XMin * scale, (float) bbox.YMax * scale),
                        new PointF((float)bbox.XMax * scale, (float) bbox.YMax * scale),
                        new PointF((float) bbox.XMax * scale, (float) bbox.YMin * scale)
                    });
            });
        }
    }
}