#include "yolo.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>


float Yolo::letterbox(
    cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool fixed_shape = true,
    bool scale_up = true) {

    cv::Size shape = image.size();

  float r = std::min(
      (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }

  int newUnpad[2]{
      (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};


  cv::Mat tmp;
  if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
    cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
  } else {
    tmp = image.clone();
  }

  float dw = new_shape.width - newUnpad[0];
  float dh = new_shape.height - newUnpad[1];


  if (!fixed_shape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  uint8_t top = uint8_t(std::round(dh - 0.1f));
  uint8_t bottom = uint8_t(std::round(dh + 0.1f));
  uint8_t left = uint8_t(std::round(dw - 0.1f));
  uint8_t right = uint8_t(std::round(dw + 0.1f));

  cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

  return 1.0f / r;
}

float* Yolo::blobFromImage(cv::Mat& img) {
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
      }
    }
  }
  return blob;
}


void Yolo::draw_objects(cv::Mat& img, float* Boxes, int* ClassIndexs, float* Scores, int* BboxNum, double t_tick, double t_start, double fps) {


    std::vector<std::string> classes_names = { "none", "mom", "Miha"};

        for (int j = 0; j < BboxNum[0]; j++) {

            std::vector<cv::Scalar> colors = {cv::Scalar(255,12,55),cv::Scalar(155,125,255), cv::Scalar(255,164,164)};

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << Scores[j];

            std::string Mystr = classes_names[ClassIndexs[j]] + " " + ss.str();

            auto font_face = cv::FONT_HERSHEY_SIMPLEX;
            auto font_scale = 0.8;
            int thickness = 1;
            int baseline = 1;
            cv::Size label_size = cv::getTextSize(Mystr, font_face, font_scale, thickness, &baseline);

            float x1  = Boxes[j * 4];
            float y1 = Boxes[j * 4 + 1];
            float width = Boxes[j * 4 + 2];
            float height = Boxes[j * 4 + 3];

            cv::Scalar classcolor = colors[ClassIndexs[j]];
            std::string cls_str = classes_names[ClassIndexs[j]];

            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x1+width,y1+height), classcolor, 2);
            cv::rectangle(img, cv::Rect(cv::Point(x1, y1), cv::Size(label_size.width, label_size.height + baseline)),
                                  classcolor, -1);


            cv::putText(img, Mystr, cv::Point(x1, y1 + label_size.height), font_face, font_scale,
                        cv::Scalar(0, 0, 0), thickness);
          }
}

Yolo::Yolo(std::string model_path) {
    ifstream ifile(model_path, ios::in | ios::binary);
    if (!ifile.good()) {
      cout << "read serialized file failed\n";
      std::abort();
    }

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
    auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("images"));
    iH = in_dims.d[2];
    iW = in_dims.d[3];
    in_size = 1;
    for (int j = 0; j < in_dims.nbDims; j++) {
    in_size *= in_dims.d[j];
    }
    auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num_dets"));
    out_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++) {
    out_size1 *= out_dims1.d[j];
    }
    auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("det_boxes"));
    out_size2 = 1;
    for (int j = 0; j < out_dims2.nbDims; j++) {
    out_size2 *= out_dims2.d[j];
    }
    auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("det_scores"));
    out_size3 = 1;
    for (int j = 0; j < out_dims3.nbDims; j++) {
    out_size3 *= out_dims3.d[j];
    }
    auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("det_classes"));
    out_size4 = 1;
    for (int j = 0; j < out_dims4.nbDims; j++) {
    out_size4 *= out_dims4.d[j];
    }
    context = engine->createExecutionContext();
    if (!context) {
    cout << "create execution context failed\n";
    std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
    cout << "allocate memory failed\n";
    std::abort();
    }
    state = cudaMalloc(&buffs[1], out_size1 * sizeof(int));
    if (state) {
    cout << "allocate memory failed\n";
    std::abort();
    }

    state = cudaMalloc(&buffs[2], out_size2 * sizeof(float));
    if (state) {
    cout << "allocate memory failed\n";
    std::abort();
    }

    state = cudaMalloc(&buffs[3], out_size3 * sizeof(float));
    if (state) {
    cout << "allocate memory failed\n";
    std::abort();
    }

    state = cudaMalloc(&buffs[4], out_size4 * sizeof(int));
    if (state) {
    cout << "allocate memory failed\n";
    std::abort();
    }

    state = cudaStreamCreate(&stream);
    if (state) {
    cout << "create stream failed\n";
    std::abort();
    }
}

void Yolo::Infer(
    int aWidth,
    int aHeight,
    int aChannel,
    unsigned char* aBytes,
    float* Boxes,
    int* ClassIndexs,
    float* Scores,
    int* BboxNum) {
  cv::Mat img(aHeight, aWidth, CV_MAKETYPE(CV_8U, aChannel), aBytes);
  cv::Mat pr_img;

  float scale = letterbox(img, pr_img, {iW, iH}, 32, {114, 114, 114}, true);
  cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
  float* blob = blobFromImage(pr_img);

  static int* num_dets = new int[out_size1];
  static float* det_boxes = new float[out_size2];
  static float* det_scores = new float[out_size3];
  static int* det_classes = new int[out_size4];

  cudaError_t state = cudaMemcpyAsync(buffs[0], &blob[0], in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
  if (state) {
    cout << "transmit to device failed\n";
    std::abort();
  }
  context->enqueueV2(&buffs[0], stream, nullptr);
  state = cudaMemcpyAsync(num_dets, buffs[1], out_size1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (state) {
    cout << "transmit to host failed \n";
    std::abort();
  }
  state = cudaMemcpyAsync(det_boxes, buffs[2], out_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (state) {
    cout << "transmit to host failed \n";
    std::abort();
  }
  state = cudaMemcpyAsync(det_scores, buffs[3], out_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  if (state) {
    cout << "transmit to host failed \n";
    std::abort();
  }
  state = cudaMemcpyAsync(
      det_classes, buffs[4], out_size4 * sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (state) {
    cout << "transmit to host failed \n";
    std::abort();
  }
  BboxNum[0] = num_dets[0];
  int img_w = img.cols;
  int img_h = img.rows;
  int x_offset = (iW * scale - img_w) / 2;
  int y_offset = (iH * scale - img_h) / 2;
  for (size_t i = 0; i < num_dets[0]; i++) {
    float x0 = (det_boxes[i * 4]) * scale - x_offset;
    float y0 = (det_boxes[i * 4 + 1]) * scale - y_offset;
    float x1 = (det_boxes[i * 4 + 2]) * scale - x_offset;
    float y1 = (det_boxes[i * 4 + 3]) * scale - y_offset;
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
    Boxes[i * 4] = x0;
    Boxes[i * 4 + 1] = y0;
    Boxes[i * 4 + 2] = x1 - x0;
    Boxes[i * 4 + 3] = y1 - y0;
    ClassIndexs[i] = det_classes[i];
    Scores[i] = det_scores[i];
  }
  delete []blob;

}

 Yolo::~Yolo() {
  cudaStreamSynchronize(stream);
  cudaFree(buffs[0]);
  cudaFree(buffs[1]);
  cudaFree(buffs[2]);
  cudaFree(buffs[3]);
  cudaFree(buffs[4]);
  cudaStreamDestroy(stream);
  context->destroy();
  engine->destroy();
  runtime->destroy();
}
