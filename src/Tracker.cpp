#include "Tracker.h"



Tracker::Tracker(size_t maximum_allowed_skipped_frames_, size_t max_trace_length_) {
    maximum_allowed_skipped_frames = maximum_allowed_skipped_frames_;
    max_trace_length = max_trace_length_;
}



void Tracker::TrackFrame(const std::vector<cv::Rect>& regions,
                        const std::vector<std::string>& labels,
                        cv::Mat& Frame)
{

    if (m_trackers.empty())
    {
        for (size_t i = 0; i < regions.size(); ++i)
        {
            m_trackers.push_back(boost::make_shared<ObjectTracker>(regions[i],
                                                                   labels[i],
                                                                  cv::Scalar(rand()%255,rand()%255, rand()%255),
                                                                 maximum_allowed_skipped_frames));
        }
    }

    if(m_trackers.empty())
        return;
    size_t N = m_trackers.size();
    size_t M = regions.size();
    std::vector<int> assignment(N, -1);
    std::vector<float> Cost(N * M);

    LocalAssociate(assignment, regions, labels,Cost);


    for (size_t i = 0; i < assignment.size(); i++)
    {
        if (assignment[i] != -1)
        {
            m_trackers[i]->Update(regions[assignment[i]], true, max_trace_length, m_prevFrame, Frame);
        }
        else if(!IsOutsideROI(Frame,m_trackers[i]->m_predictionRect))
        {
            m_trackers[i]->Update(cv::Rect(), false, max_trace_length, m_prevFrame, Frame);
        }
        else
        {
            m_trackers.erase(m_trackers.begin() + i);
            assignment.erase(assignment.begin() + i);
            i--;
        }
    }

    UpdateObjectState(assignment);

    Frame.copyTo(m_prevFrame);

    /***********
     std::cout<<"--------------shwo tracks--------------- "<<std::endl;
     for(int i=0;i<static_cast<int>(m_trackers.size());i++) {
         std::cout<<m_trackers[i]->m_showID<<" hypothesis:"<<m_trackers[i]->m_hypotheseFrames<<" soft:"<<m_trackers[i]->m_softFrames;
         std::cout<<" hard:"<<m_trackers[i]->m_hardFrames<<" skip:"<<m_trackers[i]->m_skipFrames << std::endl;
     }
     ************/

}


void Tracker::LocalAssociate(std::vector<int>& assignment,
                              const std::vector<cv::Rect>& regions,
                              const std::vector<std::string>& labels,
                              std::vector<float>& Cost)
{
    size_t N = m_trackers.size();
    size_t M = regions.size();
    for (size_t i = 0; i < m_trackers.size(); i++)
    {
            for (size_t j = 0; j < regions.size(); j++)
            {
                    auto dist = m_trackers[i]->Calciou(regions[j]);
                    Cost[i + j * N] = dist;
            }
    }
    ///进行匈牙利匹配
    AssignmentProblemSolver APS;
    APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);

    for (size_t i = 0; i < assignment.size(); i++)
    {
        // hypothese状态评估是否是目标，新出现的目标需要持续5帧
        if(m_trackers[i]->m_state==objectstate::hypothese) {
            if (assignment[i] != -1 && Cost[i + assignment[i] * N] < 0.42) {
                //you can not delete trackers here because it will affect the following code by N
                m_trackers[i]->m_hypotheseFrames++;
            } else {
                m_trackers[i]->m_state=objectstate::todelete;
            }

        }
        // hypothese状态之后是softobject状态，softobject状态是判断是否是消失的旧目标的重现
        if(m_trackers[i]->m_state==objectstate::softobject) {
            m_trackers[i]->m_softFrames++;
            if (assignment[i] != -1 && Cost[i + assignment[i] * N] < 0.42) {
                m_trackers[i]->m_skipFrames=0;
            } else {
                assignment[i] = -1;
                m_trackers[i]->m_skipFrames++;
            }
        }
        // hardobject 非常确定的目标
        if(m_trackers[i]->m_state==objectstate::hardobject) {
            m_trackers[i]->m_hardFrames++;
            if (assignment[i] != -1 && Cost[i + assignment[i] * N] < 0.42) {
                m_trackers[i]->m_skipFrames=0;
            } else {
                assignment[i] = -1;
                m_trackers[i]->m_skipFrames++;
            }
        }
    }
    ///删除错检目标
    for (size_t i = 0; i < assignment.size(); i++)
    {
      if(m_trackers[i]->m_state==objectstate::todelete)
      {
          m_trackers.erase(m_trackers.begin()+i);
          assignment.erase(assignment.begin()+i);
          i--;
      }
    }
    // 为没有匹配上的detection生成新的目标
    for (uint detection_i = 0; detection_i < regions.size(); ++detection_i) {
        auto result=find(assignment.begin(), assignment.end(), detection_i);
        if (result!= assignment.end())
                continue;
        srand(time(0));
        m_trackers.push_back(boost::make_shared<ObjectTracker>(regions[detection_i],
                                                    labels[detection_i],
                                                    cv::Scalar(rand()%255,rand()%255, rand()%255),
                                                    maximum_allowed_skipped_frames));
    }
}

void Tracker::UpdateObjectState(std::vector<int>& assignment)
{
    for (int i = 0; i < static_cast<int>(m_trackers.size()); i++)
    {
        if (m_trackers[i]->m_skipFrames > maximum_allowed_skipped_frames) {
            m_trackers[i]->m_skipFrames=0;
            m_trackers[i]->m_softFrames=0;
            m_trackers.erase(m_trackers.begin() + i);
            assignment.erase(assignment.begin()+i);
            i--;
            continue;
        }
        if(m_trackers[i]->m_hypotheseFrames > 5) {
            m_trackers[i]->m_state = objectstate::softobject;
            m_trackers[i]->m_hypotheseFrames = 0;

        }
        if(m_trackers[i]->m_softFrames >= 10) {
            m_trackers[i]->m_state = objectstate::hardobject;
            m_trackers[i]->m_softFrames = 0;
        }
        if (m_trackers[i]->m_state == objectstate::softobject) {
            m_soft_index.push_back(i);
        }

    }
}

bool Tracker::IsOutsideROI(const cv::Mat& mask, const cv::Rect_<float>& r)
{
        auto IsOutsideROI = [] (const cv::Mat& mask, const cv::Point2f& p)
        {
                int x = (int) p.x;
                int y = (int) p.y;
                bool result =  x >= (mask.cols-10) || y >= (mask.rows-10) || x <= 20|| y <= 20 ; //ShortCircuit evaluation so no out of bound access
                return result;
        };
       cv::Point p1(r.x, r.y);
       cv::Point p2(r.x+r.width, r.y);
       cv::Point p3(r.x, r.y+r.height);
       cv::Point p4(r.x+r.width, r.y+r.height);
       bool flag=IsOutsideROI(mask, p1) || IsOutsideROI(mask, p2) || IsOutsideROI(mask, p3) || IsOutsideROI(mask, p4);
       return flag;
}


std::string IntToString(const int& nNumber) {
    std::stringstream strStream;
    std::string strBuf;
    strStream << nNumber;
    strStream >> strBuf;
    return strBuf;
}


void Tracker::Draw(cv::Mat& frame)
{
    for (const auto& track : m_trackers)
    {
        if (track-> m_state != hardobject) {
            continue;
        }
        cv::Rect r = track->m_predictionRect;

        cv::rectangle(frame, r, cv::Scalar(255,10,255), 1, CV_AA);

        cv::putText(frame,track->m_label, cv::Point(r.x-5, r.y+r.height+4), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0),1,8, false);

        if(track->m_showID == -1)
        {
            track->m_showID = m_ID;
            m_ID++;
        }

        std::string showID = IntToString(track->m_showID);

        cv::putText(frame, showID, cv::Point(r.x-5, r.y), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0),2,8, false);

        for (size_t j = 0; j < track->m_trace.Getsize() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track->m_trace.GetTracePoint(j);
            const TrajectoryPoint& pt2 = track->m_trace.GetTracePoint(j+1);

            cv::Rect predictionRect1 = pt1.m_predictionRect;
            cv::Point p1 = cv::Point(predictionRect1.x + predictionRect1.width/2, predictionRect1.y + predictionRect1.height);

            cv::Rect predictionRect2 = pt2.m_predictionRect;
            cv::Point p2 = cv::Point(predictionRect2.x + predictionRect2.width/2, predictionRect2.y + predictionRect2.height);
            cv::line(frame,p1,p2,track->m_color, 1, CV_AA);
        }
    }

}



