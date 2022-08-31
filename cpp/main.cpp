//https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/cpp/end2end/main.cpp 

#include "yolo.h"
using namespace std;

int main(int argc, char** argv){


        std::string model_path = "/your/path/to/yolov7-tiny-nms.trt";
        

        std::string video_path = "/path/to/video.mp4";

        float* Boxes = new float[4000];
        float* Scores = new float[3000];
        int* BboxNum = new int[1];
        int* ClassIndexs = new int[1000];

        Yolo yolo(model_path);

        int fourcc = cv::VideoWriter::fourcc('M','P','4','V');
        cv::VideoCapture cap(0, cv::CAP_V4L2);
//        cv::VideoCapture cap(video_path);


        auto MyVideoSize = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter myvideo("/home/mih/Видео/yoloVideo/MyTestYolo31Aug.mp4",fourcc,100, MyVideoSize);
        double t_tick = 0;
        double fps = 0;

        cv::Mat img;
        while (cap.isOpened())
        {
            cap >> img;
            double t_start = cv::getTickCount();
            if(img.empty()){
                std::cout << "Can't read frame or end of file " << std::endl;
                break;

            }

            yolo.Infer(img.cols, img.rows, img.channels(), img.data, Boxes, ClassIndexs, Scores, BboxNum);

            yolo.draw_objects(img, Boxes, ClassIndexs, Scores, BboxNum, t_tick, t_start, fps);

            t_tick = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();

            fps = 1/t_tick;

            cv::putText(img, cv::format("fps :%.3f", fps), cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0,255,0), 2, 5);

            myvideo.write(img);

            cv::imshow("result", img);
            if (cv::waitKey(1)==27)
                        break;
        }

        cap.release();
        myvideo.release();

        delete []Boxes;
        delete []BboxNum;
        delete []Scores;
        delete []ClassIndexs;
        return 0;
}
