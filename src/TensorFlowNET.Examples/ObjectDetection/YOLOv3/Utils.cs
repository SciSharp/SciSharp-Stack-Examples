using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using OpenCvSharp;

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

        public static NDArray imread(string image_path)
        {
            using (var mat = Cv2.ImRead(image_path))
                return MatToNdarray<byte>(mat);
        }

        private static NDArray MatToNdarray<T>(Mat mat) 
            where T : struct 
        {
            switch (typeof(T).Name)
            {
                case "Byte":
                    {
                        var data = new Vec3b[mat.Total()];
                        //mat.GetArray(0, 0, data);
                        return Vec3bToNdarray(data, mat.Height, mat.Width, mat.Channels());
                    }
                case "Int32":
                    unsafe
                    {
                        var data = new Vec3i[mat.Total()];
                        mat.ForEachAsVec3i((value, position) => data[*position] = *value);
                        return Vec3iToNdarray(data, mat.Height, mat.Width, mat.Channels());
                    }
                case "Single":
                    unsafe
                    {
                        var data = new Vec3d[mat.Total()];
                        //mat.GetArray(0, 0, data);
                        throw new NotImplementedException("");
                    }
                default:
                    throw new NotImplementedException("");
            }
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

        private static NDArray Vec3iToNdarray(Vec3i[] data, int height, int width, int channels)
        {
            var buffer = new int[data.Length * channels];
            int i = 0;
            foreach (var d in data)
            {
                buffer[i++] = d.Item0;
                buffer[i++] = d.Item1;
                buffer[i++] = d.Item2;
            }

            return np.array(buffer).reshape(height, width, channels);
        }

        private static NDArray Vec3fToNdarray(Vec3f[] data, int height, int width, int channels)
        {
            var buffer = new float[data.Length * channels];
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
            for (var i = 0; i < data.Length; i++)
            {
                data[i].Item0 = buffer[i * 3];
                data[i].Item1 = buffer[i * 3 + 1];
                data[i].Item2 = buffer[i * 3 + 2];
            }
            return data;
        }

        private static Mat NdarrayToMat(NDArray image)
        {
            var src = new Mat(image.shape[0], image.shape[1], MatType.CV_8UC3);
            //src.SetArray(0, 0, NdarrayToVec3b(image, image.shape[0], image.shape[1], image.shape[2]));
            return src;
        }

        public static (NDArray, NDArray) image_preporcess(NDArray image, int[] target_size, NDArray gt_boxes = null)
        {
            var src = NdarrayToMat(image);
            var dst = new Mat();
            Cv2.CvtColor(src, dst, ColorConversionCodes.BGR2RGB);

            var (ih, iw) = (target_size[0], target_size[1]);
            var (h, w) = (image.shape[0], image.shape[1]);

            var scale = Math.Min(iw / (w + 0f), ih / (h + 0f));
            var (nw, nh) = (Convert.ToInt32(scale * w), Convert.ToInt32(scale * h));

            var image_resized_mat = dst.Resize(new Size(nw, nh));
            var image_resized = MatToNdarray<byte>(image_resized_mat);

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
                foreach (var nd in gt_boxes.GetNDArrays())
                {
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(0) * scale) + dw, 0);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(2) * scale) + dw, 2);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(1) * scale) + dh, 1);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(3) * scale) + dh, 3);
                }
                
                return (image_paded, gt_boxes);
            }
        }
    }
}
