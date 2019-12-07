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
    int GetFFTWindowSize() const;

private:
    // framerate
    double m_fs = 0;
    cv::Mat m_magI;
    int m_FFTWindowSize;

    size_t m_fcount = 0;
    double m_cutoff = 0;
    double m_windowSize = 0; // sec
    double m_pulse = 0;
    Iir::Butterworth::LowPass<10> m_butt;
    cv::Mat m_C;
	cv::Mat m_mean;
	cv::Mat m_std;
    cv::Mat m_U;
    cv::Mat m_prevU;
    cv::Mat n_sigmas;
    cv::Mat m_prevSigmas;
    cv::Mat m_R;
    cv::Mat m_S;
    cv::Mat m_backprojectedSR;
    cv::Mat m_p_block;
    cv::Mat m_block;

    cv::Mat m_FFTblock;

    void pushRow(cv::Mat& bufMat, cv::Mat& row);
    void pushValToBlock (cv::Mat& block, float Val);
    void getFFT(cv::Mat& input, int WindowSize, cv::Mat& magI);
};
