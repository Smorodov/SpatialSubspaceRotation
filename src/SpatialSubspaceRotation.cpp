#include "SpatialSubspaceRotation.h"

///
/// \brief SpatialSubspaceRotation::SpatialSubspaceRotation
/// \param fs
/// \param cutoff
/// \param windowSize
///
SpatialSubspaceRotation::SpatialSubspaceRotation(double fs, double cutoff, double windowSize)
{
    m_fcount = 0;
    m_cutoff = cutoff; // cutoff for lowpass filter
    m_fs = fs; // framerate [Hz]
    m_windowSize = windowSize; // window for statistical processing [sec]

    // for FFT computation
    m_FFTWindowSize = cvRound(windowSize*fs);
    m_FFTblock = cv::Mat::zeros(this->m_FFTWindowSize, 1, CV_32FC1);
    // output variable
    m_pulse = 0;
    // setup our lowpass filter
    m_butt.setup(fs, cutoff);
    //butt.setup(fs, cutoff, 1);
    m_U = cv::Mat::zeros(3, 3, CV_64FC1);
    m_prevU = cv::Mat::zeros(3, 3, CV_64FC1);
    n_sigmas = cv::Mat::zeros(3, 1, CV_64FC1);
    m_prevSigmas = cv::Mat::zeros(3, 1, CV_64FC1);
    m_R = cv::Mat::zeros(1, 2, CV_64FC1);
    m_S = cv::Mat::zeros(1, 2, CV_64FC1);
    m_backprojectedSR = cv::Mat();
    m_block = cv::Mat::zeros(m_FFTWindowSize, 3, CV_64FC1);
}

///
/// \brief SpatialSubspaceRotation::Step
/// \param values
/// \return
///
double SpatialSubspaceRotation::Step(cv::Mat& values)
{
    m_C = (values.t() * values) / values.rows;

    n_sigmas.copyTo(m_prevSigmas);
    m_U.copyTo(m_prevU);
    cv::eigen(m_C, n_sigmas, m_U);
    m_U = m_U.t();

    if (m_fcount > 0)
    {
        // rotation between the skin vector and orthonormal plane
        m_R = m_U.col(0).t() * m_prevU.colRange(1, 3);
        // scale change
        m_S.at<double>(0) = sqrt(n_sigmas.at<double>(0) / m_prevSigmas.at<double>(1));
        m_S.at<double>(1) = sqrt(n_sigmas.at<double>(0) / m_prevSigmas.at<double>(2));
        m_backprojectedSR = m_S.mul(m_R) * m_prevU.colRange(1, 3).t();
        pushRow(m_block, m_backprojectedSR);
    }

    if (m_fcount >= m_windowSize * m_fs)
    {
        cv::meanStdDev(m_block.col(0), m_mean, m_std);
        double std1 = m_std.at<double>(0);
        cv::meanStdDev(m_block.col(1), m_mean, m_std);
        double std2 = m_std.at<double>(0);
        double sigma = std1 / std2;
        //double sigma = std2 / std1;
        m_p_block = m_block.col(0) - sigma * m_block.col(1);
        m_pulse = m_butt.filter(m_p_block.at<double>(0) - mean(m_p_block)[0]);
    }
    ++m_fcount;
    // My humble modification :)
    m_pulse = tanh(200*m_pulse)/100.0;
    pushValToBlock(m_FFTblock, static_cast<float>(m_pulse));
    if (m_fcount >= m_FFTWindowSize)
    {
        getFFT(m_FFTblock, m_FFTWindowSize, m_magI);
    }
    return m_pulse;
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
void SpatialSubspaceRotation::getFFT(cv::Mat& input, int WindowSize, cv::Mat& magI)
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
    if (!m_magI.empty())
    {
        cv::normalize(m_magI, spectline, 0, 255, cv::NORM_MINMAX);
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
    return m_fs;
}

///
/// \brief SpatialSubspaceRotation::GetFFTWindowSize
/// \return
///
int SpatialSubspaceRotation::GetFFTWindowSize() const
{
    return m_FFTWindowSize;
}
