#include "SpatialSubspaceRotation.h"

///
/// \brief SpatialSubspaceRotation::SpatialSubspaceRotation
/// \param fs
/// \param cutoff
/// \param windowSize
///
SpatialSubspaceRotation::SpatialSubspaceRotation(double fs, double cutoff, double windowSize)
{
    f = 0;
    this->cutoff = cutoff; // cutoff for lowpass filter
    this->fs = fs; // framerate [Hz]
    this->windowSize = windowSize; // window for statistical processing [sec]

    // for FFT computation
    this->FFTWindowSize = windowSize*fs;
    this->FFTblock = cv::Mat::zeros(this->FFTWindowSize, 1, CV_32FC1);
    // output variable
    pulse = 0;
    // setup our lowpass filter
    butt.setup(fs, cutoff);
    //butt.setup(fs, cutoff, 1);
    U = cv::Mat::zeros(3, 3, CV_64FC1);
    U_prev = cv::Mat::zeros(3, 3, CV_64FC1);
    Sigmas = cv::Mat::zeros(3, 1, CV_64FC1);
    Sigmas_prev = cv::Mat::zeros(3, 1, CV_64FC1);
    R = cv::Mat::zeros(1, 2, CV_64FC1);
    S = cv::Mat::zeros(1, 2, CV_64FC1);
    SR_backprojected = cv::Mat();
    block = cv::Mat::zeros(windowSize * fs, 3, CV_64FC1);
}

///
/// \brief SpatialSubspaceRotation::Step
/// \param values
/// \return
///
double SpatialSubspaceRotation::Step(cv::Mat& values)
{
    C = (values.t() * values) / values.rows;

    Sigmas.copyTo(Sigmas_prev);
    U.copyTo(U_prev);
    cv::eigen(C, Sigmas, U);
    U = U.t();

    if (f > 0)
    {
        // rotation between the skin vector and orthonormal plane
        R = U.col(0).t() * U_prev.colRange(1, 3);
        // scale change
        S.at<double>(0) = sqrt(Sigmas.at<double>(0) / Sigmas_prev.at<double>(1));
        S.at<double>(1) = sqrt(Sigmas.at<double>(0) / Sigmas_prev.at<double>(2));
        SR_backprojected = S.mul(R) * U_prev.colRange(1, 3).t();
        pushRow(block, SR_backprojected);
    }

    if (f >= windowSize * fs)
    {
        cv::meanStdDev(block.col(0), m, std);
        double std1 = std.at<double>(0);
        cv::meanStdDev(block.col(1), m, std);
        double std2 = std.at<double>(0);
        double sigma = std1 / std2;
        //double sigma = std2 / std1;
        p_block = block.col(0) - sigma * block.col(1);
        pulse = butt.filter(p_block.at<double>(0) - mean(p_block)[0]);
    }
    ++f;
    // My humble modification :)
    pulse = tanh(200*pulse)/100.0;
    pushValToBlock(FFTblock, pulse);
    if (f >= FFTWindowSize)
    {
        getFFT(FFTblock, FFTWindowSize, magI);
    }
    return pulse;
}

///
/// \brief SpatialSubspaceRotation::pushRow
/// \param bufMat
/// \param row
///
void SpatialSubspaceRotation::pushRow(cv::Mat& bufMat, cv::Mat& row)
{
    cv::Mat tmp = bufMat.rowRange(1, bufMat.rows).clone();
    tmp.copyTo(bufMat.rowRange(0, bufMat.rows - 1));
    row.copyTo(bufMat.row(bufMat.rows - 1));
}

///
/// \brief SpatialSubspaceRotation::pushValToBlock
/// \param block
/// \param Val
///
void SpatialSubspaceRotation::pushValToBlock (cv::Mat& block, float Val)
{
    cv::Mat tmp = block.rowRange(1, block.rows).clone();
    tmp.copyTo(block.rowRange(0, block.rows - 1));
    block.at<float>(block.rows-1)=Val;
}

///
/// \brief SpatialSubspaceRotation::getFFT
/// \param input
/// \param WindowSize
/// \param magI
///
void SpatialSubspaceRotation::getFFT(cv::Mat& input, size_t WindowSize, cv::Mat& magI)
{
    // Calculation of FFT
    // http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
    cv::Mat plane = input.clone();
    cv::Mat planes[] = { cv::Mat_<float>(plane), cv::Mat::zeros(input.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix
    cv::split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::pow(planes[0],2, magI);
    // Cut the input into half
    magI = magI(cv::Rect(0, 0, 1, WindowSize / 2));
}

///
/// \brief SpatialSubspaceRotation::GetSpectLine
/// \return
///
cv::Mat SpatialSubspaceRotation::GetSpectLine() const
{
    cv::Mat spectline;
    if (!magI.empty())
    {
        cv::normalize(magI, spectline, 0, 255, cv::NORM_MINMAX);
        spectline.convertTo(spectline, CV_8UC1);
    }
    return spectline;
}

///
/// \brief SpatialSubspaceRotation::GetFrameRate
/// \return
///
double SpatialSubspaceRotation::GetFrameRate() const
{
    return fs;
}

///
/// \brief SpatialSubspaceRotation::GetFFTWindowSize
/// \return
///
size_t SpatialSubspaceRotation::GetFFTWindowSize() const
{
    return FFTWindowSize;
}
