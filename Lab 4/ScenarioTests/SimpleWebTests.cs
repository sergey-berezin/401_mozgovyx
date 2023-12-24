using Microsoft.AspNetCore.Mvc.Testing;
using System.Net.Http.Json;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats;
using System.Drawing;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp.Formats.Jpeg;

namespace ScenarioTests
{
    public class ScenarioWebTests : IClassFixture<WebApplicationFactory<Program>>
    {
        private readonly WebApplicationFactory<Program> _applicationFactory;

        public ScenarioWebTests(WebApplicationFactory<Program> applicationFactory)
        {
            _applicationFactory = applicationFactory;
        }

        [Fact]
        public async Task TestFakeImage()
        {
            var client = _applicationFactory.CreateClient();
            string strFakeImage = "IAmFakeBase64Image";
            var t = await client.PostAsJsonAsync("api/segmentation", strFakeImage);

            Assert.Equal(System.Net.HttpStatusCode.BadRequest, t.StatusCode);
        }

        [Fact]
        public async Task TestEmptyImage()
        {
            var client = _applicationFactory.CreateClient();
            string strEmptyImage = "";
            var t = await client.PostAsJsonAsync("api/segmentation", strEmptyImage);

            Assert.Equal(System.Net.HttpStatusCode.BadRequest, t.StatusCode);
        }

        [Fact]
        public async Task TestNullString()
        {
            var client = _applicationFactory.CreateClient();
            string? strNull = null;
            var t = await client.PostAsJsonAsync("api/segmentation", strNull);

            Assert.Equal(System.Net.HttpStatusCode.BadRequest, t.StatusCode);
        }

        [Fact]
        public async Task TestExistingImage()
        {
            var client = _applicationFactory.CreateClient();
            var image = Image.Load<Rgb24>("../../../aeroplane.jpg");
            using (MemoryStream ms = new MemoryStream())
            {
                image.Save(ms, JpegFormat.Instance);
                byte[] pixels = ms.ToArray();
                var t = await client.PostAsJsonAsync("api/segmentation", Convert.ToBase64String(pixels));
                Assert.Equal(System.Net.HttpStatusCode.OK, t.StatusCode);
            }

        }
    }
}