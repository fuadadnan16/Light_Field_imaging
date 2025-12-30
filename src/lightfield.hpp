#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct LightField {
    int lensletSize = 16;     // N in N x N angular grid
    int U = 16, V = 16;
    int S = 0, T = 0;         // spatial size per subaperture
    bool ready = false;

    // subViews[v][u] is CV_32FC3, size SxT
    std::vector<std::vector<cv::Mat>> subViews;

    void buildFromMosaic(const cv::Mat& mosaicBGR_or_RGB);
};

// Refocus: shift-and-sum all sub-aperture views by depth d
cv::Mat refocusShiftSum(const LightField& lf, float d);

// Aperture sweep: average only sub-views within radius r (in angular coords)
cv::Mat apertureAverage(const LightField& lf, float radius);

// Build a tiled mosaic (U x V tiles) for visualization
cv::Mat makeSubApertureMosaic(const LightField& lf, int tileW = 220);

// Build focal stack for multiple d values
std::vector<cv::Mat> buildFocalStack(const LightField& lf,
                                    float dMin, float dMax, int steps);

// All-in-focus via sharpness-weighted fusion (per your equations)
cv::Mat allInFocusFromStack(const std::vector<cv::Mat>& focalStack,
                            float sigmaLow = 1.0f, float sigmaSharp = 2.0f);

// Depth-from-refocus map (optional; returns CV_32F normalized 0..1)
cv::Mat depthFromRefocus(const std::vector<cv::Mat>& focalStack,
                         float dMin, float dMax);
