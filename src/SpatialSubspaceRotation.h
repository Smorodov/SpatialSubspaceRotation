#pragma once

#include <opencv2/opencv.hpp>
#include "Iir.h"

///
/// \brief The SpatialSubspaceRotation class
///
class SpatialSubspaceRotation
{
public:
    SpatialSubspaceRotation(double fs, double cutoff, double windowSize);
    ~SpatialSubspaceRotation() = default;

    double Step(cv::Mat& values);

    cv::Mat GetSpectLine() const;
    double GetFrameRate() const;
    size_t GetFFTWindowSize() const;

private:
    // framerate
    double fs = 0;
    cv::Mat magI;
    size_t FFTWindowSize;

    size_t f;
    double cutoff;
    double windowSize; // sec
    double pulse;
    Iir::Butterworth::LowPass<10> butt;
    cv::Mat C;
    cv::Mat m, std;
    cv::Mat U;
    cv::Mat U_prev;
    cv::Mat Sigmas;
    cv::Mat Sigmas_prev;
    cv::Mat R;
    cv::Mat S;
    cv::Mat SR_backprojected;
    cv::Mat p_block;
    cv::Mat block;

    cv::Mat FFTblock;

    void pushRow(cv::Mat& bufMat, cv::Mat& row);
    void pushValToBlock (cv::Mat& block, float Val);
    void getFFT(cv::Mat& input, size_t WindowSize, cv::Mat& magI);
};
