#include "ObjectTracker.h"


float ObjectTracker::CalcDist(const cv::Point_<float>& pt)
{
    cv::Point_<float> predictionPoint = (m_predictionRect.br()+m_predictionRect.tl())/2;
    cv::Point_<float> diff = predictionPoint - pt;
    return sqrtf(diff.x * diff.x + diff.y * diff.y);
}

float ObjectTracker::Calciou(const cv::Rect& r) const
{
    cv::Rect rr(m_predictionRect);
    float intArea = (r & rr).area();
    float unionArea = r.area() + rr.area() - intArea;
    return 1 - intArea / unionArea;
}



///使用匹配上的detection去更新跟踪器的目标框
void ObjectTracker::Update(const cv::Rect& region,
                           bool dataCorrect,
                           size_t max_trace_length,
                           const cv::Mat& prevFrame,
                           const cv::Mat& currFrame) {
    ///更新 m_predictionRect
    RectUpdate(region, dataCorrect, prevFrame, currFrame);
    m_trace.push_back_trace(m_predictionRect);
    // kcf预测失败
    if (!dataCorrect && m_maxVal < 0.5) {
        m_skipFrames=maximum_allowed_skipped_frames+1;
        if(m_CVtracker || !m_CVtracker.empty())
            m_CVtracker.release();
    }
    if (m_trace.Getsize() > max_trace_length) {
        m_trace.pop_front_trace(m_trace.Getsize() - max_trace_length);
    }
    m_maxVal = 1;
}


///更新
void ObjectTracker::RectUpdate(const cv::Rect& region,
                               bool dataCorrect,
                               const cv::Mat& prevFrame,
                               const cv::Mat& currFrame)
{
    if(dataCorrect) {
        m_predictionRect = region;
    } else {
        ///这里判断该跟踪器的kcf预测有没有被初始化
        if (!m_CVtracker || m_CVtracker.empty()) {
            CreateKCFTracker();
            m_CVtracker->init(prevFrame,m_predictionRect);

        }
        cv::Rect2d newRect;
        // m_maxVal is
        if (m_CVtracker->updateKCF(currFrame, newRect,m_maxVal)) {
            cv::Rect prect(cvRound(newRect.x), cvRound(newRect.y), cvRound(newRect.width), cvRound(newRect.height));
            std::cout << "kcf  m_maxVal: " << m_maxVal << std::endl;
            m_predictionRect = prect;
        }
    }
}

///创建KCF
void ObjectTracker::CreateKCFTracker()
{
    cv::TrackerKCF::Params params;
    params.compressed_size = 1;
    params.desc_pca = cv::TrackerKCF::CN;
    params.desc_npca = cv::TrackerKCF::CN;
    params.resize = true;
    m_CVtracker = cv::TrackerKCF::createTracker(params);
}
