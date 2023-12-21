using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using YoloPackage;


namespace WebApiController.Controllers
{
    [Route("api/segmentation")]
    [ApiController]
    public class HomeController : Controller
    {
        public HomeController() { }

        [HttpPost]
        public async Task<IActionResult> Post([FromBody] string strImage)
        {
            if (strImage == null || strImage.Length == 0)
            {
                return BadRequest("Controller got an empty string");
            }
            try
            {
                byte[] bytePixels = Convert.FromBase64String(strImage);
                Image<Rgb24> image;
                using (MemoryStream ms = new MemoryStream(bytePixels))
                {
                    image = Image.Load<Rgb24>(ms);
                }
                var cts = new CancellationTokenSource();
                var segmentation = await YoloService.ProcessImage(image, cts.Token);
                var result = segmentation.BoundingBoxes.Select(x =>
                    new ImagePresentation(
                        ProcessingTools.CutBoundingBox(image, x),
                        YoloService.Labels[x.Class],
                        x.Confidence
                    )
                ).ToList();
                return Ok(result);
            }
            catch (Exception e)
            {
                return BadRequest(e.Message);
            }
        }
    }
}
