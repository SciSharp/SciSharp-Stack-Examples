using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
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
        int warmup_periods;
        string time;
        float moving_ave_decay;
        int max_bbox_per_scale;
        int steps_per_period;

        Dataset trainset, testset;

        Config cfg;

        Tensor input_data;
        Tensor label_sbbox;
        Tensor label_mbbox;
        Tensor label_lbbox;
        Tensor true_sbboxes;
        Tensor true_mbboxes;
        Tensor true_lbboxes;
        Tensor trainable;

        YOLOv3 model;
        IVariableV1[] net_var;
        Tensor giou_loss, conf_loss, prob_loss;
        RefVariable global_step;
        Tensor learn_rate;
        Tensor loss;
        List<IVariableV1> first_stage_trainable_var_list;
        Operation train_op_with_frozen_variables;
        Operation train_op_with_all_variables;
        Operation train_op;
        Saver loader;
        Saver saver;
        float train_step_loss;
        double global_step_val;
        #endregion

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "YOLOv3",
                Enabled = false,
                IsImportingGraph = true
            };

        public bool Run()
        {
            PrepareData();
            Train();

            return true;
        }

        public override void Train()
        {
            var graph = Config.IsImportingGraph ? ImportGraph() : BuildGraph();

            /* for debug only
            tf.train.export_meta_graph(filename: "yolov3-debug.meta");
            var json = JsonConvert.SerializeObject(graph._nodes_by_name.Select(x => x.Value).ToArray(), Formatting.Indented);
            File.WriteAllText($"YOLOv3/nodes-{(Config.IsImportingGraph ? "right" : "wrong")}.json", json);
            */

            var config = new ConfigProto { AllowSoftPlacement = true };
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
            }
        }

        public override Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            tf_with(tf.name_scope("define_input"), scope =>
            {
                input_data = tf.placeholder(dtype: tf.float32, name: "input_data");
                label_sbbox = tf.placeholder(dtype: tf.float32, name: "label_sbbox");
                label_mbbox = tf.placeholder(dtype: tf.float32, name: "label_mbbox");
                label_lbbox = tf.placeholder(dtype: tf.float32, name: "label_lbbox");
                true_sbboxes = tf.placeholder(dtype: tf.float32, name: "sbboxes");
                true_mbboxes = tf.placeholder(dtype: tf.float32, name: "mbboxes");
                true_lbboxes = tf.placeholder(dtype: tf.float32, name: "lbboxes");
                trainable = tf.placeholder(dtype: tf.@bool, name: "training");
            });

            tf_with(tf.name_scope("define_loss"), scope =>
            {
                model = new YOLOv3(cfg, input_data, trainable);
                net_var = tf.global_variables();
                (giou_loss, conf_loss, prob_loss) = model.compute_loss(
                                                    label_sbbox, label_mbbox, label_lbbox,
                                                    true_sbboxes, true_mbboxes, true_lbboxes);
                loss = giou_loss + conf_loss + prob_loss;
            });

            Tensor global_step_update = null;
            tf_with(tf.name_scope("learn_rate"), scope =>
            {
                global_step = tf.Variable(1.0, dtype: tf.float64, trainable: false, name: "global_step");
                
                var warmup_steps = tf.constant(warmup_periods * steps_per_period,
                                        dtype: tf.float64, name: "warmup_steps");
                var train_steps = tf.constant((first_stage_epochs + second_stage_epochs) * steps_per_period,
                                        dtype: tf.float64, name: "train_steps");

                learn_rate = tf.cond(
                    pred: global_step < warmup_steps,
                    true_fn: delegate
                    {
                        return global_step / warmup_steps * learn_rate_init;
                    },
                    false_fn: delegate
                    {
                        return learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) *
                            (1 + tf.cos(
                                (global_step - warmup_steps) / (train_steps - warmup_steps) * Math.PI));
                    }
                );

                global_step_update = tf.assign_add(global_step, 1.0f) as Tensor;
            });

            Operation moving_ave = null;
            tf_with(tf.name_scope("define_weight_decay"), scope =>
            {
                var emv = tf.train.ExponentialMovingAverage(moving_ave_decay);
                var vars = tf.trainable_variables().Select(x => (RefVariable)x).ToArray();
                moving_ave = emv.apply(vars);
            });

            tf_with(tf.name_scope("define_first_stage_train"), scope =>
            {
                first_stage_trainable_var_list = new List<IVariableV1>();
                foreach (var var in tf.trainable_variables())
                {
                    var var_name = var.Op.name;
                    var var_name_mess = var_name.Split('/');
                    if (new[] { "conv_sbbox", "conv_mbbox", "conv_lbbox" }.Contains(var_name_mess[0]))
                        first_stage_trainable_var_list.Add(var);
                }

                var adam = tf.train.AdamOptimizer(learn_rate);
                var first_stage_optimizer = adam.minimize(loss, var_list: first_stage_trainable_var_list);
                tf_with(tf.control_dependencies(tf.get_collection<Operation>(tf.GraphKeys.UPDATE_OPS).ToArray()), delegate
                {
                    tf_with(tf.control_dependencies(new ITensorOrOperation[] { first_stage_optimizer, global_step_update }), delegate
                    {
                        tf_with(tf.control_dependencies(new[] { moving_ave }), delegate
                        {
                            train_op_with_frozen_variables = tf.no_op();
                        });
                    });
                });
            });

            tf_with(tf.name_scope("define_second_stage_train"), delegate
            {
                var second_stage_trainable_var_list = tf.trainable_variables().ToList();
                var adam = tf.train.AdamOptimizer(learn_rate);
                var second_stage_optimizer = adam.minimize(loss, var_list: second_stage_trainable_var_list);
                tf_with(tf.control_dependencies(tf.get_collection<Operation>(tf.GraphKeys.UPDATE_OPS).ToArray()), delegate
                {
                    tf_with(tf.control_dependencies(new ITensorOrOperation[] { second_stage_optimizer, global_step_update }), delegate
                    {
                        tf_with(tf.control_dependencies(new[] { moving_ave }), delegate
                        {
                            train_op_with_all_variables = tf.no_op();
                        });
                    });
                });
            });

            tf_with(tf.name_scope("loader_and_saver"), delegate
            {
                loader = tf.train.Saver(net_var);
                saver = tf.train.Saver(tf.global_variables(), max_to_keep: 10);
            });

            tf_with(tf.name_scope("summary"), delegate
            {
                tf.summary.scalar("learn_rate", learn_rate);
                tf.summary.scalar("giou_loss", giou_loss);
                tf.summary.scalar("conf_loss", conf_loss);
                tf.summary.scalar("prob_loss", prob_loss);
                tf.summary.scalar("total_loss", loss);
            });

            return graph;
        }

        public override Graph ImportGraph()
        {
            loader = tf.train.import_meta_graph(Path.Combine(Config.Name, "yolov3.meta"));
            var graph = tf.get_default_graph();

            train_op_with_frozen_variables = graph.OperationByName("define_first_stage_train/NoOp");
            train_op_with_all_variables = graph.OperationByName("define_second_stage_train/NoOp");
            loss = graph.get_tensor_by_name("define_loss/add_1:0");
            input_data = graph.get_tensor_by_name("define_input/input_data:0");
            label_sbbox = graph.get_tensor_by_name("define_input/label_sbbox:0");
            label_mbbox = graph.get_tensor_by_name("define_input/label_mbbox:0");
            label_lbbox = graph.get_tensor_by_name("define_input/label_lbbox:0");
            true_sbboxes = graph.get_tensor_by_name("define_input/sbboxes:0");
            true_mbboxes = graph.get_tensor_by_name("define_input/mbboxes:0");
            true_lbboxes = graph.get_tensor_by_name("define_input/lbboxes:0");
            trainable = graph.get_tensor_by_name("define_input/training:0");
            global_step = graph.OperationByName("learn_rate/global_step");

            saver = tf.train.Saver(tf.global_variables(), max_to_keep: 10);

            return graph;
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
            warmup_periods = cfg.TRAIN.WARMUP_EPOCHS;
            DateTime now = DateTime.Now;
            time = $"{now.Year}-{now.Month}-{now.Day}-{now.Hour}-{now.Minute}-{now.Minute}";
            moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY;
            max_bbox_per_scale = 150;
            trainset = new Dataset("train", cfg);
            testset = new Dataset("test", cfg);
            steps_per_period = trainset.Length;
        }
    }
}
