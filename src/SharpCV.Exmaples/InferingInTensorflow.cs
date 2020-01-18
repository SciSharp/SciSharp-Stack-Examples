using System;
using System.Collections.Generic;
using System.Text;
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
            /*var net = cv2.dnn.readNetFromTensorflow(@"D:\mask_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb",
                @"D:\mask_rcnn_inception_v2_coco_2018_01_28\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
            
            var image = cv2.imread(@"D:\SciSharp\SciSharp-Stack-Examples\data\images\cars.jpg");
            var blob = cv2.dnn.blobFromImage(image, 1.0, size: (300, 300), swapRB: true, crop: false);

            net.setInput(blob);
            var output = net.forward();*/

            return true;
        }
    }
}
