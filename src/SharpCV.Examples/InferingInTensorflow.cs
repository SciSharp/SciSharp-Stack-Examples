using NumSharp;
using static SharpCV.Binding;

namespace SharpCV.Exmaples
{
    /// <summary>
    /// https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
    /// </summary>
    internal class InferingInTensorflow
    {
        public bool Run()
        {
            var net = cv2.dnn.readNetFromTensorflow(@"D:\mask_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb",
                @"D:\mask_rcnn_inception_v2_coco_2018_01_28\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
            var img = cv2.imread(@"D:\SciSharp\SciSharp-Stack-Examples\data\images\cars.jpg");
            var rows = img.shape[0];
            var cols = img.shape[1];
            var blob = cv2.dnn.blobFromImage(img, 1.0, size: (300, 300), swapRB: true, crop: false);
            net.setInput(blob);
            NDArray output = net.forward();

            foreach (var detection in output[0, 0, Slice.All, Slice.All].GetNDArrays())
            {
                var score = detection.GetSingle(2);
                if (score > 0.3)
                {
                    var left = detection.GetSingle(3) * cols;
                    var top = detection.GetSingle(4) * rows;
                    var right = detection.GetSingle(5) * cols;
                    var bottom = detection.GetSingle(6) * rows;
                    cv2.rectangle(img, ((int)left, (int)top), ((int)right, (int)bottom), (23, 230, 210), thickness: 2);
                }
            }

            cv2.imshow("img", img);
            cv2.waitKey();
            return true;
        }
    }
}
