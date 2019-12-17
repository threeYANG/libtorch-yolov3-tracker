#pragma once
#include "opencv2/opencv.hpp"
#include <boost/make_shared.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "ObjectTracker.h"
#include "HungarianAlg.h"
// ----------------------------------------------------------------------
class Tracker
{
public:
    Tracker(size_t maximum_allowed_skipped_frames_,
            size_t max_trace_length_ );

    ~Tracker()= default;

    void TrackFrame(const std::vector<cv::Rect>& regions,
                  const std::vector<std::string>& labels,
                  cv::Mat& frame);

    void Draw(cv::Mat& frame);



private:

    void LocalAssociate(std::vector<int>& assignment,
                       const std::vector<cv::Rect>& regions,
                       const std::vector<std::string>& labels,
                       std::vector<float>& Cost);

    void UpdateObjectState(std::vector<int>& assignment);

    bool IsOutsideROI(const cv::Mat& mask, const cv::Rect_<float>& r);

    std::vector<boost::shared_ptr<ObjectTracker>> m_trackers;

    size_t maximum_allowed_skipped_frames;
    size_t max_trace_length;
    cv::Mat m_prevFrame;
    std::vector<int> m_soft_index;

    int m_ID = 0;
};
