#include "opencv2/opencv.hpp"
#include <windows.h>
#include <algorithm>
#include <set>
#include "Iir.h"

using namespace cv;
using namespace std;
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
void generateSkinMap(cv::Mat& src, cv::Mat& dst)
{
    cv::cvtColor(src, dst, cv::COLOR_BGR2YCrCb);
    cv::Mat mask;
    cv::inRange(dst, Scalar(0, 133, 98), Scalar(255, 177, 142), mask);
    dst = mask.clone();
}
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
void extractPixels(cv::Mat& img_src, cv::Mat& values)
{
    cv::Mat bin;
    size_t pixelsInImage = (img_src.rows * img_src.cols);
    generateSkinMap(img_src, bin);
    size_t NPx = cv::countNonZero(bin);
    values = cv::Mat::zeros(NPx, 3, CV_64FC1);
    size_t n = 0;
    for (int j = 0; j < img_src.rows; ++j)
    {
        for (int k = 0; k < img_src.cols; ++k)
        {
            if (bin.at<uchar>(j, k) > 0)
            {
                values.at<double>(n, 0) = img_src.at<Vec3b>(j, k)[2]; // R
                values.at<double>(n, 1) = img_src.at<Vec3b>(j, k)[1]; // G
                values.at<double>(n, 2) = img_src.at<Vec3b>(j, k)[0]; // B
                ++n;
            }
        }
    }
}
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
class SSR
{
public:
    // framerate
    double fs;
    cv::Mat magI;
    size_t FFTWindowSize;

    SSR(double fs, double cutoff, double windowSize)
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

    //cv::Mat values(NPixels, 3, CV_64FC1);
    double Step(cv::Mat& values)
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

private:
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
    //------------------------------------------
    //
    //------------------------------------------
    void pushRow(cv::Mat& bufMat, cv::Mat& row)
    {
        Mat tmp = bufMat.rowRange(1, bufMat.rows).clone();
        tmp.copyTo(bufMat.rowRange(0, bufMat.rows - 1));
        row.copyTo(bufMat.row(bufMat.rows - 1));
    }
    //------------------------------------------
    //
    //------------------------------------------
    void pushValToBlock (cv::Mat& block, float Val)
    {
        Mat tmp = block.rowRange(1, block.rows).clone();
        tmp.copyTo(block.rowRange(0, block.rows - 1));
        block.at<float>(block.rows-1)=Val;
    }
    void getFFT(cv::Mat& input, size_t WindowSize, cv::Mat& magI)
    {
        // Calculation of FFT
        // http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
        Mat plane = input.clone();
        Mat planes[] = { Mat_<float>(plane), Mat::zeros(input.size(), CV_32F) };
        Mat complexI;
        merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
        dft(complexI, complexI);            // this way the result may fit in the source matrix
        split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
        pow(planes[0],2, magI);
        // Cut the input into half
        magI = magI(Rect(0, 0, 1, WindowSize / 2));
    }
};

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    cv::Mat img_src;
    cv::Mat plot = Mat::zeros(600, 1024, CV_8UC3);
    VideoCapture cap("./../face.avi");

    SSR ssr(10, 2, 5);
    double pulse = 0;
    double pulse_prev = 0;
    size_t f = 0;
    int k = 0;

    while (k != 27)
    {
        cap >> img_src;
        double start = static_cast<double>(cv::getTickCount());
        cout << " f : " << std::to_string(f) << endl;
        if (img_src.empty())
        {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        cv::Mat values;
        extractPixels(img_src, values);

        if (values.empty())
        {
            continue;
        }

        pulse_prev = pulse;
        pulse = ssr.Step(values);
 
        if (f >= plot.cols)
        {
            f = 0;
            plot.setTo(Scalar::all(0));
        }

        if (!ssr.magI.empty())
        {
            Mat spectline;
            normalize(ssr.magI, ssr.magI, 0, 255, cv::NORM_MINMAX);
            ssr.magI.convertTo(spectline, CV_8UC1);
            
            applyColorMap(spectline, spectline, COLORMAP_JET);

            spectline.copyTo(plot(Rect(f, 0, 1, ssr.FFTWindowSize/2)));
        }



        float y1 = pulse_prev * 1000 + 300;
        float y2 = pulse * 1000 + 300;
        cv::line(plot, Point(f-1, y1), Point(f, y2), Scalar::all(255), 1);
        {
            imshow("img_src", img_src);
            imshow("plot", plot);
            k = waitKey(5);
        }
        ++f;

        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency() * 1000;

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t2;
        std::chrono::duration<double> time_span = std::chrono::duration<double>(0);

        while (time_span.count() < (1000.0 / ssr.fs - time_cost) / 1000.0)
        {
            t2 = std::chrono::steady_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        }
        double end2 = static_cast<double>(cv::getTickCount());
        time_cost = (end2 - start) / cv::getTickFrequency() * 1000;
        std::cout << "time cost: " << time_cost << "ms" << std::endl;
    }
return 0;
}
