/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Binding;
using static SharpCV.Binding;
using SharpCV;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/shimat/opencvsharp/wiki/Capturing-Video
    /// </summary>
    public class YoloCoco : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "YoloCoco";
        public bool IsImportingGraph { get; set; } = true;

        int input_size = 416;
        int num_classes = 80;

        string[] return_elements = new[] 
        { 
            "input/input_data:0", 
            "pred_sbbox/concat_2:0", 
            "pred_mbbox/concat_2:0", 
            "pred_lbbox/concat_2:0" 
        };

        Tensor[] return_tensors;
        
        public bool Run()
        {
            PrepareData();

            var graph = ImportGraph();

            using (var sess = new Session(graph))
                Predict(sess);

            return true;       
        }

        public void PrepareData()
        {
            // download video
            string url = "https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/docs/images/road.mp4";
            Web.Download(url, Name, "road.mp4");
        }

        public Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();

            var bytes = File.ReadAllBytes(Path.Combine(Name, "yolov3_coco.pb"));
            var graphDef = GraphDef.Parser.ParseFrom(bytes);
            return_tensors = tf.import_graph_def(graphDef, return_elements: return_elements)
                .Select(x => x as Tensor)
                .ToArray();

            return graph;
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            // Opens MP4 file (ffmpeg is probably needed)
            var vid = cv2.VideoCapture(Path.Combine(Name, "road.mp4"));

            int sleepTime = (int)Math.Round(1000 / 24.0);

            var (loaded, frame) = vid.read();
            while (loaded)
            {
                var frame_size = (frame.shape[0], frame.shape[1]);
                var image_data = image_preporcess(frame, (input_size, input_size));
                image_data = image_data[np.newaxis, Slice.Ellipsis];

                var (pred_sbbox, pred_mbbox, pred_lbbox) = sess.run((return_tensors[1], return_tensors[2], return_tensors[3]),
                    (return_tensors[0], image_data));

                var pred_bbox = np.concatenate((np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))), axis: 0);

                var bboxes = postprocess_boxes(pred_bbox, frame_size, input_size, 0.3f);

                // cv2.imshow("result", frame);
                // cv2.waitKey(sleepTime);

                (loaded, frame) = vid.read();
            }
        }

        private NDArray image_preporcess(Mat image, (int, int) target_size)
        {
            image = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2RGB);
            var (ih, iw) = target_size;
            var (h, w) = (image.shape[0] + 0.0f, image.shape[1] + 0.0f);
            var scale = min(iw / w, ih / h);
            var (nw, nh) = ((int)Math.Round(scale * w), (int)Math.Round(scale * h));
            var image_resized = cv2.resize(image, (nw, nh));
            var image_padded = np.full((ih, iw, 3), fill_value: 128.0f);
            var (dw, dh) = ((iw - nw) / 2, (ih - nh) / 2);
            image_padded[new Slice(dh, nh + dh), new Slice(dw, nw + dw), Slice.All] = image_resized;
            image_padded = image_padded / 255;

            return image_padded;
        }

        private NDArray postprocess_boxes(NDArray pred_bbox, (int, int) org_img_shape, float input_size, float score_threshold)
        {
            var valid_scale = new[] { 0, np.inf };
            pred_bbox = np.array(pred_bbox);

            var pred_xywh = pred_bbox[Slice.All, new Slice(0, 4)];
            var pred_conf = pred_bbox[Slice.All, 4];
            var pred_prob = pred_bbox[Slice.All, new Slice(5)];

            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var pred_coor = np.concatenate((pred_xywh[Slice.All, new Slice(stop: 2)] - pred_xywh[Slice.All, new Slice(2)] * 0.5f,
                                        pred_xywh[Slice.All, new Slice(stop: 2)] + pred_xywh[Slice.All, new Slice(2)] * 0.5f), axis: -1);
            
            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            var (org_h, org_w) = org_img_shape;
            var resize_ratio = min(input_size / org_w, input_size / org_h);
            var dw = (input_size - resize_ratio * org_w) / 2;
            var dh = (input_size - resize_ratio * org_h) / 2;

            pred_coor[Slice.All, new Slice(0, step: 2)] = 1.0 * (pred_coor[Slice.All, new Slice(0, step: 2)] - dw) / resize_ratio;
            pred_coor[Slice.All, new Slice(1, step: 2)] = 1.0 * (pred_coor[Slice.All, new Slice(1, step: 2)] - dh) / resize_ratio;

            // (3) clip some boxes those are out of range
            /*pred_coor = np.concatenate((np.maximum(pred_coor[Slice.All, new Slice(stop: 2)], np.array(new[] { 0, 0 })),
             np.minimum(pred_coor[Slice.All, new Slice(2)], np.array(new[] { org_w - 1, org_h - 1 }))), axis: -1);

            var invalid_mask = np.arange(1);// np.logical_or(pred_coor[Slice.All, 0] > pred_coor[Slice.All, 2], pred_coor[Slice.All, 1] > pred_coor[Slice.All, 3]);
            pred_coor[invalid_mask] = 0;

            // (4) discard some invalid boxes
            var bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[Slice.All, new Slice(2, 4)] - pred_coor[Slice.All, new Slice(0, 2)], axis: -1));
            var scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]));

            // (5) discard some boxes with low scores
            NDArray coors;
            var classes = np.argmax(pred_prob, axis: -1);
            var scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes];
            var score_mask = scores > score_threshold;
            var mask = np.logical_and(scale_mask, score_mask);
            (coors, scores, classes) = (pred_coor[mask], scores[mask], classes[mask]);

            return np.concatenate(new[] { coors, scores[Slice.All, np.newaxis], classes[Slice.All, np.newaxis] }, axis: -1);*/
            return pred_bbox;
        }

        private NDArray nms(NDArray bboxes, float iou_threshold, float sigma = 0.3f, string method = "nms")
        {
            return bboxes;
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
