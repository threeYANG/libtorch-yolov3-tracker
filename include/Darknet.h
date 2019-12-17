/*******************************************************************************
* 
* Author : walktree
* Email  : walktree@gmail.com
*
* A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. 
* It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~
*
*******************************************************************************/

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

struct Darknet : torch::nn::Module {

public:

	Darknet(const char *conf_file, int input_net_size, int class_num, float nms_thresh, float conf_thresh, torch::Device *device);

	map<string, string>* get_net_info();

	void load_weights(const char *weight_file);

	torch::Tensor transform(const cv::Mat& img);

	torch::Tensor forward(torch::Tensor x);

    void TensorBoxToVector(torch::Tensor result,  int width, int height,
                           std::vector<cv::Rect>& regions, std::vector<std::string>& labels,
                           const std::vector<string>& labels_name);



private:

    int input_net_size_;

    int class_num_;

    int nms_thresh_;

    int conf_thresh_;

	torch::Device *_device;

	vector<map<string, string>> blocks;

	torch::nn::Sequential features;

	vector<torch::nn::Sequential> module_list;

    // load YOLOv3 
    void load_cfg(const char *cfg_file);

    void create_modules();

    int get_int_from_cfg(map<string, string> block, string key, int default_value);

    string get_string_from_cfg(map<string, string> block, string key, string default_value);

    /**
	 *  对预测数据进行筛选
	 */
    torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf = 0.4);
};