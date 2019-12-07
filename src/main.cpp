#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include "SpatialSubspaceRotation.h"

///
/// \brief generateSkinMap
/// \param src
/// \param dst
///
void generateSkinMap(cv::Mat& src, cv::Mat& dst)
{
    cv::cvtColor(src, dst, cv::COLOR_BGR2YCrCb);
    cv::Mat mask;
    cv::inRange(dst, cv::Scalar(0, 133, 98), cv::Scalar(255, 177, 142), mask);
    dst = mask.clone();
}

///
/// \brief extractPixels
/// \param img_src
/// \param values
///
void extractPixels(cv::Mat& img_src, cv::Mat& values)
{
    cv::Mat bin;
    size_t pixelsInImage = (img_src.rows * img_src.cols);
    generateSkinMap(img_src, bin);
    int NPx = cv::countNonZero(bin);
    values = cv::Mat::zeros(NPx, 3, CV_64FC1);
    int n = 0;
    for (int j = 0; j < img_src.rows; ++j)
    {
        for (int k = 0; k < img_src.cols; ++k)
        {
            if (bin.at<uchar>(j, k) > 0)
            {
                values.at<double>(n, 0) = img_src.at<cv::Vec3b>(j, k)[2]; // R
                values.at<double>(n, 1) = img_src.at<cv::Vec3b>(j, k)[1]; // G
                values.at<double>(n, 2) = img_src.at<cv::Vec3b>(j, k)[0]; // B
                ++n;
            }
        }
    }
}

///
/// \brief main
/// \param argc
/// \param argv
/// \return
///
int main(int argc, char** argv)
{
    cv::Mat img_src;
    cv::Mat plot = cv::Mat::zeros(600, 1024, CV_8UC3);
    
	std::string fileName = "0";
	if (argc > 1)
	{
		fileName = argv[1];
	}
	cv::VideoCapture cap;
	if (fileName.length() == 1)
		cap.open(atoi(fileName.c_str()));
	else
			cap.open(fileName);

    SpatialSubspaceRotation ssr(10, 2, 5);
    double pulse = 0;
    double pulse_prev = 0;
    int f = 0;
    int k = 0;

    while (k != 27)
    {
        cap >> img_src;
        double start = static_cast<double>(cv::getTickCount());
        std::cout << " f : " << std::to_string(f) << std::endl;
        if (img_src.empty())
        {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
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
            plot.setTo(cv::Scalar::all(0));
        }

        cv::Mat spectline = ssr.GetSpectLine();
        if (!spectline.empty())
        {
            cv::applyColorMap(spectline, spectline, cv::COLORMAP_JET);
            spectline.copyTo(plot(cv::Rect(f, 0, 1, ssr.GetFFTWindowSize()/2)));
        }

        int y1 = cvRound(pulse_prev * 1000 + 300);
        int y2 = cvRound(pulse * 1000 + 300);
        cv::line(plot, cv::Point(f-1, y1), cv::Point(f, y2), cv::Scalar::all(255), 1);
        {
            cv::imshow("img_src", img_src);
            cv::imshow("plot", plot);
            k = cv::waitKey(5);
        }
        ++f;

        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency() * 1000;

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration<double>(0);

        while (time_span.count() < (1000.0 / ssr.GetFrameRate() - time_cost) / 1000.0)
        {
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1);
        }
        double end2 = static_cast<double>(cv::getTickCount());
        time_cost = (end2 - start) / cv::getTickFrequency() * 1000;
        std::cout << "pulse = " << pulse << ", time cost: " << time_cost << "ms" << std::endl;
    }
    return 0;
}
