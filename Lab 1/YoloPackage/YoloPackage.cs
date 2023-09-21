using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloPackage
{
    public class YoloService
    {
        private const string ModelURL = "https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx";
        private InferenceSession? Session = null;
        private SemaphoreSlim SessionLock = new SemaphoreSlim(1, 1);

        private async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> Forward(List<NamedOnnxValue> inputs) {
            await SessionLock.WaitAsync();
            var outputs = Session.Run(inputs);
            SessionLock.Release();
            return outputs;
        }
    }
}