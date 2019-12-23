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
using OpenCvSharp;
using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/shimat/opencvsharp/wiki/Capturing-Video
    /// </summary>
    public class DetectFromVideo : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Detect From Video";
        public bool IsImportingGraph { get; set; } = true;

        public bool Run()
        {
            PrepareData();

            // Opens MP4 file (ffmpeg is probably needed)
            VideoCapture capture = new VideoCapture("road.mp4");

            int sleepTime = (int) Math.Round(1000 / capture.Fps);

            using (Window window = new Window("capture"))
            using (Mat image = new Mat()) // Frame image buffer
            {
                // When the movie playback reaches end, Mat.data becomes NULL.
                while (true)
                {
                    capture.Read(image); // same as cvQueryFrame
                    if (image.Empty())
                        break;

                    window.ShowImage(image);
                    Cv2.WaitKey(sleepTime);
                } 
            }     

            return true;       
        }

        public void PrepareData()
        {
            // download video
            // https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/docs/images/road.mp4
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
