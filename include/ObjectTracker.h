#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <string>

///目标状态: hypothese一般持续0-5帧，过滤错检
///         softobject一般持续10帧，这个状态下跟踪器需要和其他的跟踪器匹配
///         hardobject确定是目标
///         todelete需要删除

enum objectstate {
     hypothese=0,
     softobject=1,
     hardobject=2,
     todelete
};

///轨迹结构
struct TrajectoryPoint
{
        TrajectoryPoint()= default;

        explicit TrajectoryPoint(const cv::Rect& predictionRect){m_predictionRect = predictionRect;}

        cv::Rect m_predictionRect;
};



class Trace
{
public:

    TrajectoryPoint GetValue(int i) {
        return m_motion[i];
    }
    int Getsize() const {
        return m_motion.size();
    }

    TrajectoryPoint GetTracePoint(int j) {
        return m_motion[j];
    }

	void push_back_trace(const cv::Rect& predictionRect)
	{
        m_motion.emplace_back(predictionRect);
	}
	void pop_front_trace(size_t count)
	{
	    if (count < m_motion.size())
	    {
		  m_motion.erase(m_motion.begin(), m_motion.begin() + count);
	    }
	    else
	    {
		  m_motion.clear();
	    }
	}

private:
        std::deque<TrajectoryPoint> m_motion;
};




///每个目标其实就是一个跟踪器，即ObjectTracker
class ObjectTracker
{
public:
    ObjectTracker( const cv::Rect& region,
            std::string label,
            cv::Scalar color,
            size_t skipframes):
            m_showID(-1),
            maximum_allowed_skipped_frames(skipframes)
          {
            m_label = std::move(label);
            m_color = std::move(color);
            m_state=objectstate::hypothese;
            m_hypotheseFrames=0;
            m_softFrames=0;
            m_hardFrames=0;
            m_skipFrames=0;

            m_predictionRect = region;

            m_maxVal = 1;
          }
	~ObjectTracker()= default;

public:
    float CalcDist(const cv::Point_<float>& pt);

    float Calciou(const cv::Rect& r) const;

    void Update(const cv::Rect& region,bool dataCorrect,size_t max_trace_length,const cv::Mat& prevFrame, const cv::Mat& currFrame);

public:
	///显示参数
	std::string m_label;
	int m_showID;
	cv::Scalar m_color;
	///状态参数
	uint m_hypotheseFrames;
	uint m_softFrames;
	uint m_hardFrames;
	uint m_skipFrames;
	objectstate m_state;
	///当前帧结果参数
	cv::Rect m_predictionRect;
	///允许KCF存活最长的帧数
    size_t maximum_allowed_skipped_frames;
    ///车辆轨迹
    Trace m_trace;

private:
    void RectUpdate(const cv::Rect& region,bool dataCorrect,const cv::Mat& prevFrame,const cv::Mat& currFrame);
    void CreateKCFTracker();

    ///KCF跟踪器
    cv::Ptr<cv::TrackerKCF> m_CVtracker;
    //KCF预测返回参数(0,1)之间
    double m_maxVal;
};


