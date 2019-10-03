using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using OpenCvSharp;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    class Utils
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

        public static NDArray imread(string image_path)
        {
            using (var mat = Cv2.ImRead(image_path))
                return MatToNdarray(mat);
        }

        private static NDArray MatToNdarray(Mat mat)
        {
            var data = new Vec3b[mat.Total()];
            mat.GetArray(0, 0, data);
            return Vec3bToNdarray(data, mat.Height, mat.Width, mat.Channels());
        }

        private static NDArray Vec3bToNdarray(Vec3b[] data, int height, int width, int channels)
        {
            var buffer = new byte[data.Length * channels];
            int i = 0;
            foreach (var d in data)
            {
                buffer[i++] = d.Item0;
                buffer[i++] = d.Item1;
                buffer[i++] = d.Item2;
            }

            return np.array(buffer).reshape(height, width, channels);
        }

        private static Vec3b[] NdarrayToVec3b(NDArray image, int height, int width, int channels)
        {
            var data = new Vec3b[height * width];
            var buffer = image.ToByteArray();
            for (var i = 0; i < data.Length; i += 3)
            {
                data[i].Item0 = buffer[i * 3];
                data[i].Item1 = buffer[i * 3 + 1];
                data[i].Item2 = buffer[i * 3 + 2];
            }
            return data;
        }

        public static (NDArray, NDArray) image_preporcess(NDArray image, int[] target_size, NDArray gt_boxes = null)
        {
            var src = new Mat(new Size(375, 500), MatType.CV_8UC3);
            src.SetArray(0, 0, NdarrayToVec3b(image, 375, 500, 3));

            var dst = new Mat();
            Cv2.CvtColor(src, dst, ColorConversionCodes.BGR2RGB);

            var data = new Vec3b[dst.Total()];
            dst.GetArray(0, 0, data);
            image = Vec3bToNdarray(data, dst.Height, dst.Width, dst.Channels());

            var (ih, iw) = (target_size[0], target_size[1]);
            var (h, w) = (image.shape[0], image.shape[1]);

            var scale = Math.Min(iw / (w + 0f), ih / (h + 0f));
            var (nw, nh) = (Convert.ToInt32(scale * w), Convert.ToInt32(scale * h));

            var input = InputArray.Create(dst);
            var image_resized_mat = OutputArray.Create(new Mat());
            Cv2.Resize(input, image_resized_mat, new Size(nw, nh));
            var image_resized = MatToNdarray(image_resized_mat.GetMat());
            var image_paded = np.full((ih, iw, 3), fill_value: 128.0);
            var (dw, dh) = (Math.Floor((iw - nw) / 2f), Math.Floor((ih - nh) / 2f));
            //image_paded[$"dh:nh + dh", $"dw: nw + dw", $":"] = image_resized;
            throw new NotImplementedException("");
        }
    }
}
