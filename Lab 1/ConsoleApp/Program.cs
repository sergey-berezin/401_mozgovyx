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
        private static CancellationTokenSource cts = new CancellationTokenSource();
        
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
                        var processingTask = YoloService.ProcessImage(image, cts.Token);

                        var filename = Path.GetFileName(arg);
                        Directory.CreateDirectory("bounding_boxes");
                        var path = $"bounding_boxes/{filename}";

                        await processingTask;
                        image = processingTask.Result.Image;
                        var boundingBoxes = processingTask.Result.BoundingBoxes;
                        var savingJpegTask = image.SaveAsJpegAsync(path.ToLower(), cts.Token);

                        objectsLock.Wait();
                        objects.AddRange(boundingBoxes.Select(
                            bb => new CSVLine(
                                arg,
                                YoloService.Labels[bb.Class],
                                (int) bb.XMin,
                                (int) bb.YMin,
                                (int) (bb.XMax - bb.XMin),
                                (int) (bb.YMax - bb.YMin)
                            )
                        ));
                        objectsLock.Release();
                        await savingJpegTask;
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"`{arg}`: {e.Message}");
                    }
                }, cts.Token);
            }));

            // Some tasks can complete their calculations, if `model.onnx` is already downloaded
            /*
            Thread.Sleep(1000);
            cts.Cancel();
            */
            
            try
            {
                await argsProcessingTask;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
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