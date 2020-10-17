using Newtonsoft.Json;
using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    /// <summary>
    /// Implementation of YOLO v3 object detector in Tensorflow
    /// https://github.com/YunYang1994/tensorflow-yolov3
    /// </summary>
    public class SampleYOLOv3 : SciSharpExample, IExample
    {
        #region args
        Dictionary<int, string> classes;
        int num_classes;
        float learn_rate_init;
        float learn_rate_end;
        int first_stage_epochs;
        int second_stage_epochs;
        int warmup_steps;
        int total_steps;
        string time;
        float moving_ave_decay;
        int max_bbox_per_scale;
        int steps_per_epoch;

        Dataset trainset, testset;

        Config cfg;

        Tensor input_tensor;
        Tensor label_sbbox;
        Tensor label_mbbox;
        Tensor label_lbbox;
        Tensor true_sbboxes;
        Tensor true_mbboxes;
        Tensor true_lbboxes;
        Tensor trainable;

        Model model;
        IVariableV1[] net_var;
        Tensor giou_loss, conf_loss, prob_loss;
        IVariableV1 global_step;
        Tensor learn_rate;
        Tensor loss;
        List<IVariableV1> first_stage_trainable_var_list;
        Operation train_op_with_frozen_variables;
        Operation train_op_with_all_variables;
        Operation train_op;
        Saver loader;
        Saver saver;
        float train_step_loss;
        IVariableV1 global_steps;
        #endregion

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "YOLOv3 (Eager)",
                Enabled = true
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            Train();

            return true;
        }

        void train_step(NDArray image_data, (NDArray, NDArray)[] target)
        {
            using var tape = tf.GradientTape();
            var pred_result = model.Apply(image_data);
        }

        public override void Train()
        {
            input_tensor = tf.keras.layers.Input((416, 416, 3));
            var yolo = new YOLOv3(cfg);
            var conv_tensors = yolo.Apply(input_tensor);

            var output_tensors = new List<Tensor>();
            foreach (var (i, conv_tensor) in enumerate(conv_tensors))
            {
                var pred_tensor = yolo.Decode(conv_tensor, i);
                output_tensors.append(conv_tensor);
                output_tensors.append(pred_tensor);
            }

            model = tf.keras.Model(input_tensor, output_tensors);
            // model.load_weights("./yolov3");

            var optimizer = tf.keras.optimizers.Adam();
            foreach (var epoch in range(1, 1 + first_stage_epochs + second_stage_epochs))
            {
                // tf.print('EPOCH %3d' % (epoch + 1))
                foreach (var (image_data, target) in trainset)
                    train_step(image_data, target);
            }

            /* for debug only
            tf.train.export_meta_graph(filename: "yolov3-debug.meta");
            var json = JsonConvert.SerializeObject(graph._nodes_by_name.Select(x => x.Value).ToArray(), Formatting.Indented);
            File.WriteAllText($"YOLOv3/nodes-{(Config.IsImportingGraph ? "right" : "wrong")}.json", json);
            */

            /*var config = new ConfigProto { AllowSoftPlacement = true };
            using (var sess = tf.Session(graph, config: config))
            {
                sess.run(tf.global_variables_initializer());
                print($"=> Restoring weights from: {cfg.TRAIN.INITIAL_WEIGHT} ... ");
                // loader.restore(sess, cfg.TRAIN.INITIAL_WEIGHT);
                first_stage_epochs = 20;

                foreach (var epoch in range(1, 1 + first_stage_epochs + second_stage_epochs))
                {
                    if (epoch <= first_stage_epochs)
                        train_op = train_op_with_frozen_variables;
                    else
                        train_op = train_op_with_all_variables;

                    int batch = 1;
                    foreach (var train_data in trainset.Items())
                    {
                        var results = sess.run(new object[] { train_op, loss, global_step },
                            (input_data, train_data[0]),
                            (label_sbbox, train_data[1]),
                            (label_mbbox, train_data[2]),
                            (label_lbbox, train_data[3]),
                            (true_sbboxes, train_data[4]),
                            (true_mbboxes, train_data[5]),
                            (true_lbboxes, train_data[6]),
                            (trainable, true));

                        (train_step_loss, global_step_val) = (results[1], results[2]);
                        //train_epoch_loss.append(train_step_loss);
                        // summary_writer.add_summary(summary, global_step_val)
                        print($"epoch {epoch} batch {batch}, train loss: {train_step_loss}");
                        batch++;
                    }

                    float test_step_loss = 0;
                    foreach (var test_data in testset.Items())
                    {
                        test_step_loss = sess.run(loss,
                            (input_data, test_data[0]),
                            (label_sbbox, test_data[1]),
                            (label_mbbox, test_data[2]),
                            (label_lbbox, test_data[3]),
                            (true_sbboxes, test_data[4]),
                            (true_mbboxes, test_data[5]),
                            (true_lbboxes, test_data[6]),
                            (trainable, false));

                        print($"test loss: {test_step_loss}");
                        // test_epoch_loss.append(test_step_loss);
                    }

                    string ckpt_file = $"./{Config.Name}/checkpoint/yolov3_test_loss={test_step_loss}.ckpt";
                    saver.save(sess, ckpt_file, global_step: epoch);
                }
            }*/
        }

        public override void PrepareData()
        {
            cfg = new Config(Config.Name);

            string dataDir = Path.Combine(Config.Name, "data");
            Directory.CreateDirectory(dataDir);

            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_classes = classes.Count;

            learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT;
            learn_rate_end = cfg.TRAIN.LEARN_RATE_END;
            first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS;
            second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS;
            DateTime now = DateTime.Now;
            time = $"{now.Year}-{now.Month}-{now.Day}-{now.Hour}-{now.Minute}-{now.Minute}";
            moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY;
            max_bbox_per_scale = 150;
            trainset = new Dataset("train", cfg);
            testset = new Dataset("test", cfg);
            steps_per_epoch = trainset.Length;

            global_steps = tf.Variable(1, trainable: false, dtype: tf.int64);
            warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch;
            total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch;
        }
    }
}
