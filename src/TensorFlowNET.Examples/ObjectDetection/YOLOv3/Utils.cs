using NumSharp;
using SharpCV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static SharpCV.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Utils
    {
        public static Dictionary<int, string> read_class_names(string file)
        {
            var classes = new Dictionary<int, string>();
            foreach (var line in File.ReadAllLines(file))
                classes[classes.Count] = line;
            return classes;
        }

        public static NDArray get_anchors(string file)
        {
            return np.array(File.ReadAllText(file).Split(',')
                .Select(x => float.Parse(x))
                .ToArray()).reshape(3, 3, 2);
        }

        public static (NDArray, NDArray) image_preporcess(Mat image, int[] target_size, NDArray gt_boxes = null)
        {
            var dst = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2RGB);

            var (ih, iw) = (target_size[0], target_size[1]);
            var (h, w) = (image.shape[0], image.shape[1]);

            var scale = Math.Min(iw / (w + 0f), ih / (h + 0f));
            var (nw, nh) = (Convert.ToInt32(scale * w), Convert.ToInt32(scale * h));

            NDArray image_resized = cv2.resize(dst, (nw, nh));

            var image_paded = np.full((ih, iw, 3), fill_value: 128.0);
            var (dw, dh) = (Convert.ToInt32(Math.Floor((iw - nw) / 2f)), Convert.ToInt32(Math.Floor((ih - nh) / 2f)));

            image_paded[new Slice(dh, nh + dh), new Slice(dw, nw + dw), new Slice()] = image_resized;
            image_paded = image_paded / 255;

            if (gt_boxes == null)
            {
                return (image_paded, gt_boxes);
            }
            else
            {
                gt_boxes[Slice.All, new Slice(0, 4, 2)] = gt_boxes[Slice.All, new Slice(0, 4, 2)] * scale + dw;
                gt_boxes[Slice.All, new Slice(1, 4, 2)] = gt_boxes[Slice.All, new Slice(1, 4, 2)] * scale + dh;

                /*foreach (var nd in gt_boxes.GetNDArrays())
                {
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(0) * scale) + dw, 0);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(2) * scale) + dw, 2);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(1) * scale) + dh, 1);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(3) * scale) + dh, 3);
                }*/
                
                return (image_paded, gt_boxes);
            }
        }
    }
}
