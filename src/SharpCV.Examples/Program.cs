namespace SharpCV.Exmaples
{
    class Program
    {
        static void Main(string[] args)
        {
            var inferingInTensorflow = new InferingInTensorflow();
            inferingInTensorflow.Run();

            var camera = new CameraCapture();
            camera.Run();
        }
    }
}
