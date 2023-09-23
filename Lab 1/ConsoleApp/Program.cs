using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloPackage;

namespace ConsoleApp
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("No arguments provided. Try `program img1 ... imgN`");
                return;
            }

            List<CSVLine> objects = new();
            SemaphoreSlim objectsLock = new SemaphoreSlim(1, 1);
            
            YoloService.Logger = new ConsoleLogger();
            var argsProcessingTask = Task.WhenAll(args.Select(arg => {
                return Task.Run(async () => {
                    try
                    {
                        var image = Image.Load<Rgb24>(arg);
                        // Getting component results
                        var result = await YoloService.ProcessImage(image);

                        // Saving image to `bounding_boxes` folder
                        var cwd = Path.GetDirectoryName(arg);
                        var filename = Path.GetFileName(arg);
                        Directory.CreateDirectory($"{cwd}/bounding_boxes");
                        var path = $"{cwd}/bounding_boxes/{filename}";
                        await result.Image.SaveAsJpegAsync(path);

                        // Getting all `.csv` lines
                        await objectsLock.WaitAsync();
                        objects.AddRange(result.BoundingBoxes.Select(
                            bb => new CSVLine(
                                arg,
                                YoloService.Labels[bb.Class],
                                Convert.ToInt32(bb.XMin),
                                Convert.ToInt32(bb.YMin),
                                Convert.ToInt32(bb.XMax - bb.XMin),
                                Convert.ToInt32(bb.YMax - bb.YMin)
                            )
                        ));
                        objectsLock.Release();
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"`{arg}`: {e.Message}");
                    }
                });
            }));
            
            /*
            // Some tasks can complete their calculations, if `model.onnx` is already downloaded
            Thread.Sleep(1000);
            YoloService.cts.Cancel();
            */
            await argsProcessingTask;

            SaveCVS(objects);
        }
        
        const string CSVPath = "bounding_boxes.csv";
        const string CSVHead = "Filename,Class,X,Y,W,H";

        private static void SaveCVS(IEnumerable<CSVLine> objects)
        {
            using (StreamWriter sw = File.CreateText(CSVPath))
            {
                sw.WriteLine(CSVHead);
                foreach (var obj in objects)
                {
                    sw.WriteLine(obj.ToString());
                }
            }
        }
    }

    public class ConsoleLogger : ILogger
    {
        public void SendMessage(string? message)
        {
            Console.WriteLine(message);
        }
    }

    public record CSVLine(string filename, string classname, int X, int Y, int W, int H)
    {
        public override string ToString()
        {
            return $"\"{filename}\",\"{classname}\",{X},{Y},{W},{H}";
        }
    }
}