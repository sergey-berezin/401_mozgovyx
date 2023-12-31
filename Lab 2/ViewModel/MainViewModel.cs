using System;
using System.Collections.Generic;
using System.Windows.Input;
using ImageProcessingLib;
using YoloPackage;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading.Tasks;
using AsyncCommand;
using System.Threading;
using System.Linq;
using System.Windows.Media.Imaging;
using System.Runtime.CompilerServices;
using System.Windows.Media;

namespace ViewModel
{
    public interface IUIServices
    {
        List<string> ExtractFiles(string folderName, string format);
        string? GetFolderName();
        void ReportError(string message);
    }

    public class DetectedImageView
    {
        #region FIELDS
        private ObjectBox BBox { get; set; }
        private Image<Rgb24> OriginalImage { get; }
        private WeakReference SelectedImageRef { get; set; }
        #endregion

        #region PUBLIC_PROPERTIES
        public BitmapSource SelectedImage
        {
            get
            {
                var selectedImage = SelectedImageRef.Target;
                if (selectedImage == null)
                {
                    var mainImage = ImageToBitmapSource(ProcessingTools.Annotate(OriginalImage, BBox));
                    SelectedImageRef = new WeakReference(mainImage);
                    return mainImage;
                }
                else
                {
                    return (BitmapSource)selectedImage;
                }
            }
        }
        public BitmapSource Image { get; }
        public string Class { get; set; }
        public double Confidence { get; set; }
        #endregion

        public DetectedImageView(SegmentedObject segmentedObject)
        {
            BBox = segmentedObject.bbox;
            OriginalImage = segmentedObject.OriginalImage;
            SelectedImageRef = new WeakReference(null);
            Image = ImageToBitmapSource(segmentedObject.BoundingBox);
            Class = segmentedObject.Class;
            Confidence = segmentedObject.Confidence;
        }

        private BitmapSource ImageToBitmapSource(Image<Rgb24> image)
        {
            byte[] pixels = new byte[image.Width * image.Height * Unsafe.SizeOf<Rgb24>()];
            image.CopyPixelDataTo(pixels);

            return BitmapFrame.Create(image.Width, image.Height, 96, 96, PixelFormats.Rgb24, null, pixels, 3 * image.Width);
        }
    }

    public class MainViewModel : ViewModelBase
    {
        #region FIELDS
        private string selectedFolder { get; set; } = string.Empty;
        private bool isModelActive { get; set; } = false;
        private CancellationTokenSource cts { get; set; }
        #endregion

        #region PUBLIC_PROPERTIES
        public string SelectedFolder
        {
            get => selectedFolder;
            set
            {
                if (value != null && value != selectedFolder)
                {
                    selectedFolder = value;
                    RaisePropertyChanged(nameof(SelectedFolder));
                }
            }
        }
        public List<DetectedImageView> DetectedImages { get; private set; }
        #endregion

        private readonly IUIServices uiServices;

        #region COMMANDS
        private void OnSelectFolder(object arg)
        {
            string? folderName = uiServices.GetFolderName();
            if (folderName == null) { return; }
            SelectedFolder = folderName;
        }
        public async Task OnRunModel(object arg)
        {
            DetectedImages.Clear();
            RaisePropertyChanged(nameof(DetectedImages));
            try
            {
                isModelActive = true;
                cts = new CancellationTokenSource();

                List<string> fileNames = uiServices.ExtractFiles(SelectedFolder, ".jpg");
                if (fileNames.Count == 0)
                {
                    uiServices.ReportError("This folder doesn't contain .jpg files");
                    return;
                }
                var tasks = fileNames.Select(arg => 
                    Task.Run(() => ProcessingTools.FindImageSegmentation(arg, cts.Token))
                ).ToList();
                
                while (tasks.Any())
                {
                    var task = await Task.WhenAny(tasks);
                    var detectedObjects = task.Result.ToList();
                    tasks.Remove(task);
                    DetectedImages = DetectedImages.Concat(
                        detectedObjects.Select(x => new DetectedImageView(x))
                    ).ToList();
                    RaisePropertyChanged(nameof(DetectedImages));
                }
            }
            catch (Exception e)
            {
                uiServices.ReportError(e.Message);
            }
            finally
            {
                isModelActive = false;
            }
        }
        public void OnRequestCancellation(object arg)
        {
            cts.Cancel();
        }
        public ICommand SelectFolderCommand { get; private set; }
        public ICommand RunModelCommand { get; private set; }
        public ICommand RequestCancellationCommand { get; private set; }
        #endregion

        public MainViewModel(IUIServices uiServices)
        {
            SelectedFolder = string.Empty;
            DetectedImages = new List<DetectedImageView>();

            this.uiServices = uiServices;
            
            SelectFolderCommand = new RelayCommand(OnSelectFolder, x => !isModelActive);
            RunModelCommand = new AsyncRelayCommand(OnRunModel, x => SelectedFolder != string.Empty && !isModelActive);
            RequestCancellationCommand = new RelayCommand(OnRequestCancellation, x => isModelActive);
        }
    }
}
