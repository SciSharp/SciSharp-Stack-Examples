using Microsoft.AspNetCore.Mvc;
using Tensorflow.NumPy;
using WebApi.Services;

namespace WebApi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly ILogger<ImageClassificationController> _logger;
        private readonly ImageClassificationService _imageClassifier;


        public ImageClassificationController(ILogger<ImageClassificationController> logger, ImageClassificationService imageClassifier)
        {
            _logger = logger;
            _imageClassifier = imageClassifier;
        }

        [HttpGet("train")]
        public IActionResult Train()
        {
            _imageClassifier.Train();
            return Ok();
        }

        [HttpGet("predict")]
        public IActionResult Predict()
        {
            // Validate incoming bytes
            /*var length = request.ContentLength;
            if (length is null)
            {
                return BadRequest("Missing Content-Length header.");
            }
            else if (length != 28 * 28)
            {
                return BadRequest("Content-Length should be exactly 28x28 bytes.");
            }

            // Read HTTP request body into buffer
            var buffer = new byte[(int)length];
            await request.Body.ReadAsync(buffer, 0, (int)length);*/

            // Give their bytes to the neural network as input
            var prediction = _imageClassifier.Predict(new byte[28 * 28]);

            // Send back what we think it is, as an array of 10 doubles
            return Ok(prediction);
        }
    }
}