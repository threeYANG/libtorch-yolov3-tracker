#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "Darknet.h"
#include "Tracker.h"


using namespace std; 
using namespace std::chrono;

const vector<string> labels_name = {"car", "others", "person"};


int main(int argc, const char* argv[])
{
    if (argc != 4) {
        std::cerr << "usage: yolo-app <cfg path>  <weight path>  <image path>\n";
        return -1;
    }

    torch::DeviceType device_type;
    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);
    // input image size for YOLO v3
    int input_net_size = 512;
    int num_class = 3;
    float nms_conf = 0.3;
    float thresh_conf = 0.2;

    Darknet net(argv[1], input_net_size, num_class, nms_conf, thresh_conf, &device);
    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_net_size);

    net.load_weights(argv[2]);
    net.to(device);
    torch::NoGradGuard no_grad;
    net.eval();

    boost::shared_ptr<Tracker> m_tracker = nullptr;
    cv::VideoCapture capture;
    if (!capture.open(argv[3]))
    {
        std::cerr << "Can't open the video: " << argv[3] << std::endl;
        return 1;
    }

    cv::Mat frame;

    bool first_frame = true;
    while(capture.read(frame)) {

        auto start = std::chrono::high_resolution_clock::now();
        auto img_tensor = net.transform(frame);
        auto output = net.forward(img_tensor);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        // It should be known that it takes longer time at first time
        std::cout << "inference taken : " << duration.count() << " ms" << endl;

        std::vector<cv::Rect> regions;
        std::vector<std::string> labels;

        net.TensorBoxToVector(output, frame.cols, frame.rows, regions, labels, labels_name);


        if (first_frame) {
            m_tracker = boost::make_shared<Tracker>(10, 2 * 25);
            first_frame = false;
        }
        m_tracker->TrackFrame(regions, labels, frame);

        m_tracker->Draw(frame);

        imshow("Demo",frame);
        cv::waitKey(10);

    }








    
    return 0;
}
