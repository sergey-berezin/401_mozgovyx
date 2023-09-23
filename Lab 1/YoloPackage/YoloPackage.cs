using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloPackage
{
    public static class YoloService
    {
        private const string ModelURL = "https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx";
        private const string ModelFilename = "model.onnx";
        private const int TargetSize = 416;
        private const int CellCount = 13;
        private const int BoxCount = 5;
        private const int ClassCount = 20;
        public static readonly string[] Labels = new string[] {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };
        private static readonly (double, double)[] Anchors = new (double, double)[] {
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52)
        };
        private static InferenceSession? Session = null;
        private static SemaphoreSlim SessionLock = new SemaphoreSlim(1, 1);
        public static ILogger? Logger = null;
        public static CancellationTokenSource cts = new CancellationTokenSource();

        private static void CheckCancellation(string? taskName = null)
        {
            if (cts.IsCancellationRequested)
            {
                if (taskName != null)
                    throw new TaskCanceledException($"{taskName} is requested to be cancelled");
                else
                    throw new TaskCanceledException();
            }
        }

        private static async Task DownloadModel()
        {
            Logger?.SendMessage($"Downloading YOLO model from {ModelURL}");
            using var client = new HttpClient();
            using var data = await client.GetStreamAsync(ModelURL);
            using var fileStream = new FileStream(ModelFilename, FileMode.OpenOrCreate);
            await data.CopyToAsync(fileStream);
            Logger?.SendMessage("Model has been downloaded");
        }

        private static async Task SetUpModel()
        {
            await SessionLock.WaitAsync();
            while (!cts.IsCancellationRequested && Session == null)
            {
                try
                {
                    Logger?.SendMessage("Starting new session");
                    Session = new InferenceSession(ModelFilename, new SessionOptions
                    {
                        // Too many warnings per millisecond
                        LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                    });
                    Logger?.SendMessage("Session has been started");
                }
                catch (Exception)
                {
                    await DownloadModel();
                }
            }
            SessionLock.Release();
        }

        private static async Task<Tensor<float>> Forward(List<NamedOnnxValue> inputs)
        {
            await SessionLock.WaitAsync();
            if (Session == null)
            {
                throw new Exception("Current session is null");
            }
            var outputs = Session.Run(inputs);
            SessionLock.Release();
            return outputs.First().AsTensor<float>();
        }

        private static List<ObjectBox> GetObjectBoxes(Tensor<float> outputs)
        {
            List<ObjectBox> objects = new();
            int cellSize = TargetSize / CellCount;

            for (var row = 0; row < CellCount; row++)
                for (var col = 0; col < CellCount; col++)
                    for (var box = 0; box < BoxCount; box++)
                    {
                        var rawX = outputs[0, (5 + ClassCount) * box, row, col];
                        var rawY = outputs[0, (5 + ClassCount) * box + 1, row, col];

                        var rawW = outputs[0, (5 + ClassCount) * box + 2, row, col];
                        var rawH = outputs[0, (5 + ClassCount) * box + 3, row, col];

                        var x = (float)((col + TensorMath.Sigmoid(rawX)) * cellSize);
                        var y = (float)((row + TensorMath.Sigmoid(rawY)) * cellSize);

                        var w = (float)(Math.Exp(rawW) * Anchors[box].Item1 * cellSize);
                        var h = (float)(Math.Exp(rawH) * Anchors[box].Item2 * cellSize);

                        var conf = TensorMath.Sigmoid(outputs[0, (5 + ClassCount) * box + 4, row, col]);

                        if (conf > 0.5)
                        {
                            var classes
                            = Enumerable
                            .Range(0, ClassCount)
                            .Select(i => outputs[0, (5 + ClassCount) * box + 5 + i, row, col])
                            .ToArray();
                            objects.Add(
                                new ObjectBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf, TensorMath.Argmax(TensorMath.Softmax(classes))));
                        }
                    }
            return objects;
        }

        private static void RemoveDuplicateBoxes(List<ObjectBox> objects)
        {
            for (int i = 0; i < objects.Count; i++)
            {
                var o1 = objects[i];
                for (int j = i + 1; j < objects.Count;)
                {
                    var o2 = objects[j];
                    if (o1.Class == o2.Class && o1.IoU(o2) > 0.6)
                    {
                        if (o1.Confidence < o2.Confidence)
                        {
                            objects[i] = o1 = objects[j];
                        }
                        objects.RemoveAt(j);
                    }
                    else
                    {
                        j++;
                    }
                }
            }
        }

        private static void Annotate(Image<Rgb24> target, IEnumerable<ObjectBox> objects)
        {
            foreach (var objbox in objects)
            {
                target.Mutate(ctx =>
                {
                    ctx.DrawPolygon(
                        Pens.Solid(Color.Blue, 2),
                        new PointF[] {
                            new PointF((float)objbox.XMin, (float)objbox.YMin),
                            new PointF((float)objbox.XMin, (float)objbox.YMax),
                            new PointF((float)objbox.XMax, (float)objbox.YMax),
                            new PointF((float)objbox.XMax, (float)objbox.YMin)
                        });

                    ctx.DrawText(
                        $"{Labels[objbox.Class]}",
                        SystemFonts.Families.First().CreateFont(16),
                        Color.Blue,
                        new PointF((float)objbox.XMin, (float)objbox.YMax));
                });
            }
        }

        public static async Task<YoloSegmentation> ProcessImage(Image<Rgb24> image)
        {
            CheckCancellation(nameof(SetUpModel));
            await SetUpModel();

            CheckCancellation("ResizeImage");
            var resized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetSize, TargetSize),
                    Mode = ResizeMode.Pad
                });
            });

            CheckCancellation("ImageToTensor");
            var input = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });
            resized.ProcessPixelRows(pa =>
            {
                for (int y = 0; y < TargetSize; y++)
                {
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < TargetSize; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });

            CheckCancellation("TensorToInputs");
            var inputs = new List<NamedOnnxValue>
            {
               NamedOnnxValue.CreateFromTensor("image", input)
            };

            CheckCancellation(nameof(Forward));
            var outputs = await Forward(inputs);

            CheckCancellation(nameof(GetObjectBoxes));
            var objects = GetObjectBoxes(outputs);
            
            CheckCancellation(nameof(RemoveDuplicateBoxes));
            RemoveDuplicateBoxes(objects);
            
            CheckCancellation(nameof(Annotate));
            Annotate(resized, objects);

            return new YoloSegmentation(resized, objects);
        }
    }

    public record ObjectBox(double XMin, double YMin, double XMax, double YMax, double Confidence, int Class)
    {
        public double IoU(ObjectBox b2) =>
            (Math.Min(XMax, b2.XMax) - Math.Max(XMin, b2.XMin)) * (Math.Min(YMax, b2.YMax) - Math.Max(YMin, b2.YMin)) /
            ((Math.Max(XMax, b2.XMax) - Math.Min(XMin, b2.XMin)) * (Math.Max(YMax, b2.YMax) - Math.Min(YMin, b2.YMin)));
    }

    public record YoloSegmentation(Image<Rgb24> Image, IEnumerable<ObjectBox> BoundingBoxes);

    public static class TensorMath
    {
        public static float Sigmoid(float value)
        {
            var e = (float)Math.Exp(value);
            return e / (1.0f + e);
        }

        public static float[] Softmax(float[] values)
        {
            var exps = values.Select(v => Math.Exp(v));
            var sum = exps.Sum();
            return exps.Select(e => (float)(e / sum)).ToArray();
        }

        public static int Argmax(float[] values)
        {
            int idx = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[idx])
                    idx = i;
            return idx;
        }
    }


    public interface ILogger
    {
        public void SendMessage(string? message);
    }
}