using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Dataset
    {
        string annot_path;
        int[] input_sizes;
        int batch_size;
        bool data_aug;
        int train_input_size;
        int[] train_input_sizes;
        NDArray train_output_sizes;
        NDArray strides;
        NDArray anchors;
        Dictionary<int, string> classes;
        int num_classes;
        int anchor_per_scale;
        int max_bbox_per_scale;
        string[] annotations;
        int num_samples;
        int num_batchs;
        int batch_count;

        public int Length = 0;

        public Dataset(string dataset_type, Config cfg)
        {
            annot_path = dataset_type == "train" ? cfg.TRAIN.ANNOT_PATH : cfg.TEST.ANNOT_PATH;
            input_sizes = dataset_type == "train" ? cfg.TRAIN.INPUT_SIZE : cfg.TEST.INPUT_SIZE;
            batch_size = dataset_type == "train" ? cfg.TRAIN.BATCH_SIZE : cfg.TEST.BATCH_SIZE;
            data_aug = dataset_type == "train" ? cfg.TRAIN.DATA_AUG : cfg.TEST.DATA_AUG;
            train_input_sizes = cfg.TRAIN.INPUT_SIZE;
            strides = np.array(cfg.YOLO.STRIDES);

            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_classes = classes.Count;
            anchors = np.array(Utils.get_anchors(cfg.YOLO.ANCHORS));
            anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE;
            max_bbox_per_scale = 150;

            annotations = load_annotations();
            num_samples = len(annotations);
            num_batchs = Convert.ToInt32(Math.Ceiling(num_samples / Convert.ToDecimal(batch_size)));
            batch_count = 0;
        }

        string[] load_annotations()
        {
            return File.ReadAllLines(annot_path);
        }

        public IEnumerable<NDArray[]> Items()
        {
            /*for (int i = 3; i < 6; i++)
            {
                var results = new List<NDArray>();
                for (int j = 0; j < 7; j++)
                    results.Add(np.load($"YOLOv3/data/npy/training/data-{i}.0-{j}.npy"));
                yield return results.ToArray();
            }*/


            train_input_size = 448;// train_input_sizes[new Random().Next(0, train_input_sizes.Length - 1)];
            train_output_sizes = train_input_size / strides;
            var batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3));
            var batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                          anchor_per_scale, 5 + num_classes));
            var batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                          anchor_per_scale, 5 + num_classes));
            var batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                          anchor_per_scale, 5 + num_classes));

            var batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4));
            var batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4));
            var batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4));

            int num = 0;
            if (batch_count < num_batchs)
            {
                while (num < batch_size)
                {
                    var index = batch_count * batch_size + num;
                    if (index >= num_samples)
                        index -= num_samples;
                    var annotation = "D:/VOC\\train/VOCdevkit/VOC2007\\JPEGImages\\000192.jpg 116,64,356,375,14";// annotations[index];
                    var (image, bboxes) = parse_annotation(annotation);
                    var (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) = preprocess_true_boxes(bboxes);

                    batch_image[num, Slice.All, Slice.All, Slice.All] = image;
                    batch_label_sbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_sbbox;
                    batch_label_mbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_mbbox;
                    batch_label_lbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_lbbox;
                    batch_sbboxes[num, Slice.All, Slice.All] = sbboxes;
                    batch_mbboxes[num, Slice.All, Slice.All] = mbboxes;
                    batch_lbboxes[num, Slice.All, Slice.All] = lbboxes;
                    num += 1;
                }
                batch_count += 1;

                /*return new[]
                {
                    batch_image,
                    batch_label_sbbox,
                    batch_label_mbbox,
                    batch_label_lbbox,
                    batch_sbboxes,
                    batch_mbboxes,
                    batch_lbboxes
                };*/
                throw new NotImplementedException("");
            }
            else
            {
                batch_count = 0;
                np.random.shuffle(annotations);
                throw new StopIteration();
            }
        }

        private (NDArray, NDArray) parse_annotation(string annotation)
        {
            var line = annotation.Split();
            var image_path = line[0];
            if (!File.Exists(image_path))
                throw new KeyError($"{image_path} does not exist ... ");
            NDArray image = Utils.imread(image_path);

            var bboxes = np.stack(line
                .Skip(1)
                .Select(box => np.array(box
                        .Split(',')
                        .Select(x => int.Parse(x))
                        .ToArray()))
                .ToArray());

            
            if (data_aug)
            {
                //(image, bboxes) = random_horizontal_flip(np.copy(image), np.copy(bboxes));
                //(image, bboxes) = random_crop(np.copy(image), np.copy(bboxes));
                //(image, bboxes) = random_translate(np.copy(image), np.copy(bboxes));
            }
            (image, bboxes) = Utils.image_preporcess(np.copy(image), new[] { train_input_size, train_input_size }, np.copy(bboxes));
            return (image, bboxes);
        }

        private(NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) preprocess_true_boxes(NDArray bboxes)
        {
            var label = range(3).Select(i => np.zeros(train_output_sizes[i], train_output_sizes[i], anchor_per_scale, 5 + num_classes)).ToArray();
            var bboxes_xywh = range(3).Select(x => np.zeros(max_bbox_per_scale, 4)).ToArray();
            var bbox_count = np.zeros(3);

            foreach(var bbox in bboxes.GetNDArrays())
            {
                var bbox_coor = bbox[":4"];
                int bbox_class_ind = bbox[4];

                var onehot = np.zeros(new Shape(num_classes), dtype: np.float32);
                onehot[bbox_class_ind] = 1.0f;
                var uniform_distribution = np.full(new Shape(num_classes), 1.0 / num_classes);
                var deta = 0.01f;
                var smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution;

                var bbox_xywh = np.concatenate(((bbox_coor["2:"] + bbox_coor[":2"]) * 0.5, bbox_coor["2:"] - bbox_coor[":2"]), axis: -1);
                var bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, Slice.All] / strides[Slice.All, np.newaxis];
                var iou = new List<NDArray>();
                var exist_positive = false;
                foreach(var i in range(3))
                {
                    var anchors_xywh = np.zeros((anchor_per_scale, 4));
                    anchors_xywh[Slice.All, new Slice(0, 2)] = np.floor(bbox_xywh_scaled[i, new Slice(0, 2)]).astype(np.int32) + 0.5;
                    anchors_xywh[Slice.All, new Slice(2, 4)] = anchors[i];

                    var iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, Slice.All], anchors_xywh);
                    iou.Add(iou_scale);
                    var iou_mask = np.array(new[] { false, false, true }).AsGeneric<bool>(); // iou_scale > 0.3;
                    if (np.any(iou_mask))
                    {
                        var floors = np.floor(bbox_xywh_scaled[i, new Slice(0, 2)]).astype(np.int32);
                        var (xind, yind) = (floors.GetInt32(0), floors.GetInt32(1));

                        label[i][yind, xind, iou_mask, Slice.All] = 0;
                        label[i][yind, xind, iou_mask, new Slice(0, 4)] = bbox_xywh;
                        label[i][yind, xind, iou_mask, new Slice(4, 5)] = 1.0f;
                        label[i][yind, xind, iou_mask, new Slice(5)] = smooth_onehot;

                        var bbox_ind = (int)(bbox_count[i] % max_bbox_per_scale);
                        bboxes_xywh[i][bbox_ind, new Slice(0, 4)] = bbox_xywh;
                        bbox_count[i] += 1;
                        exist_positive = true;
                    }
                }

                if (!exist_positive)
                {
                    throw new NotImplementedException("");
                }
            }

            var (label_sbbox, label_mbbox, label_lbbox) = (label[0], label[1], label[2]);
            var (sbboxes, mbboxes, lbboxes) = (bboxes_xywh[0], bboxes_xywh[1], bboxes_xywh[2]);

            return (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes);
        }

        private NDArray bbox_iou(NDArray boxes1, NDArray boxes2)
        {
            var boxes1_area = boxes1[Slice.Ellipsis, 2] * boxes1[Slice.Ellipsis, 3];
            var boxes2_area = boxes2[Slice.Ellipsis, 2] * boxes2[Slice.Ellipsis, 3];

            boxes1 = np.concatenate((boxes1[Slice.Ellipsis, new Slice(":2")] - boxes1[Slice.Ellipsis, new Slice("2:")] * 0.5,
                boxes1[Slice.Ellipsis, new Slice(":2")] + boxes1[Slice.Ellipsis, new Slice("2:")] * 0.5), 
                axis: -1);

            boxes2 = np.concatenate((boxes2[Slice.Ellipsis, new Slice(":2")] - boxes2[Slice.Ellipsis, new Slice("2:")] * 0.5,
                boxes2[Slice.Ellipsis, new Slice(":2")] + boxes2[Slice.Ellipsis, new Slice("2:")] * 0.5), 
                axis: -1);

            var left_up = np.maximum(boxes1[Slice.Ellipsis, new Slice(":2")], boxes2[Slice.Ellipsis, new Slice(":2")]);
            var right_down = np.minimum(boxes1[Slice.Ellipsis, new Slice("2:")], boxes2[Slice.Ellipsis, new Slice("2:")]);
            var inter_section = np.maximum(right_down - left_up, NDArray.Scalar(0.0f));
            var inter_area = inter_section[Slice.Ellipsis, 0] * inter_section[Slice.Ellipsis, 1];
            var union_area = boxes1_area + boxes2_area - inter_area;

            return inter_area / union_area;
        }
    }
}
