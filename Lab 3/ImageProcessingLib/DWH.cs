using Newtonsoft.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.CompilerServices;
using YoloPackage;


namespace ImageProcessingLib
{
    public class ImagePresentation
    {
        public string Pixels { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public List<string> Classes { get; set; }
        public List<double> Confidences { get; set; }
        public List<ObjectBox> ObjectBoxes { get; set; }
        public string Filename { get; set; }

        public ImagePresentation(IEnumerable<SegmentedObject> segmentation, string filename)
        {
            Filename = filename;
            if (segmentation == null || segmentation.Count() == 0)
            {
                Pixels = string.Empty;
                Height = 0;
                Width = 0;
                Classes = new List<string>();
                Confidences = new List<double>();
                ObjectBoxes = new List<ObjectBox>();
            }
            else
            {
                Image<Rgb24> image = segmentation.First().OriginalImage;
                byte[] bytePixels = new byte[image.Width * image.Height * Unsafe.SizeOf<Rgb24>()];
                image.CopyPixelDataTo(bytePixels);
                Pixels = Convert.ToBase64String(bytePixels);
                Height = image.Height;
                Width = image.Width;
                if (Height == 0 || Width == 0)
                {
                    throw new Exception($"Height: {Height}, Width: {Width}");
                }
                Classes = segmentation.Select(x => x.Class).ToList();
                Confidences = segmentation.Select(x => x.Confidence).ToList();
                ObjectBoxes = segmentation.Select(x => x.bbox).ToList();
            }
            Filename = filename;
        }

        public List<SegmentedObject> ToSegmentedObjectList()
        {
            List<SegmentedObject> segmented = new List<SegmentedObject>();
            byte[] bytePixels = Convert.FromBase64String(Pixels);
            Image<Rgb24> image = Image.LoadPixelData<Rgb24>(bytePixels, Width, Height);
            for (int i = 0; i < Classes.Count(); i++)
            {
                segmented.Add(
                    new SegmentedObject(
                        image,
                        Classes[i],
                        Confidences[i],
                        ProcessingTools.CutBoundingBox(image, ObjectBoxes[i]),
                        ObjectBoxes[i]
                    )
                );
            }
            return segmented;
        }
    }

    public class JsonStorage
    {
        public string Path { get; private set; }
        public List<ImagePresentation> Images { get; private set; }

        public JsonStorage(string path = "storage.json")
        {
            Path = path;
            Images = new List<ImagePresentation>();
        }

        public void Erase()
        {
            Images.Clear();
            if (File.Exists(Path))
                File.Delete(Path);
        }

        public void Load()
        {
            if (!File.Exists(Path))
                return;
            var images = JsonConvert.DeserializeObject<List<ImagePresentation>>(File.ReadAllText(Path));
            if (images != null)
                Images = images;
            else
                Images = new List<ImagePresentation>();
        }

        public void Save()
        {
            string tmpPath = Path + ".tmp";
            string serialized = JsonConvert.SerializeObject(Images, Formatting.Indented);
            using (StreamWriter writer = new StreamWriter(tmpPath))
            {
                writer.WriteLine(serialized);
            }
            if (File.Exists(tmpPath))
            {
                File.Delete(Path);
                File.Copy(tmpPath, Path);
                if (File.Exists(Path))
                    File.Delete(tmpPath);
            }
        }

        public void AddImage(ImagePresentation image)
        {
            bool imageExists = false;
            foreach (var existing in Images)
            {
                if (image.Pixels == existing.Pixels || image.Filename == existing.Filename)
                {
                    imageExists = true;
                    break;
                }
            }
            if (!imageExists && image.Height > 0 && image.Width > 0)
                Images.Add(image);
        }
    }
}
