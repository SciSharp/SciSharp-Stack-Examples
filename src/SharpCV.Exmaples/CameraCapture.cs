using static SharpCV.Binding;

namespace SharpCV.Exmaples
{
    internal class CameraCapture
    {
        public bool Run()
        {
            var vid = cv2.VideoCapture(0);

            var (loaded, frame) = vid.read();
            while (loaded)
            {
                cv2.imshow("result", frame);
                cv2.waitKey(30);
                (loaded, frame) = vid.read();
            }

            return true;
        }
    }
}
