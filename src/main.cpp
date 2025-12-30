#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "lightfield.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

// ---------------- File dialog ----------------
static std::string openFileDialog() {
#ifdef _WIN32
    char fileName[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter =
        "Images\0*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff\0";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    if (GetOpenFileNameA(&ofn)) return std::string(fileName);
    return "";
#else
    std::string p;
    std::cout << "Enter image path: ";
    std::getline(std::cin, p);
    return p;
#endif
}

// ---------------- Helpers ----------------
static cv::Mat toDisplay8U(const cv::Mat& img01) {
    cv::Mat out;
    cv::Mat clipped;
    cv::min(cv::max(img01, 0.0f), 1.0f, clipped);
    clipped.convertTo(out, CV_8U, 255.0);
    return out;
}

static void drawTitle(cv::Mat& img, const std::string& text) {
    cv::putText(img, text, {10, 25},
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        {255,255,255}, 2);
}

// ---------------- Main ----------------
int main() {
    std::string path = openFileDialog();
    if (path.empty()) return 0;

    cv::Mat mosaic = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (mosaic.empty()) return 1;

    LightField lf;
    lf.lensletSize = 16;
    lf.buildFromMosaic(mosaic);
    if (!lf.ready) return 1;

    // Precompute static data
    cv::Mat mosaicLF = makeSubApertureMosaic(lf, 240);

    // Sliders
    int depthSlider = 60;
    int aperSlider  = 8;

    const int tileW = 480;
    const int tileH = 360;
    const int headerH = 40;
    const int footerH = 60;

    const int canvasW = tileW * 2;
    const int canvasH = headerH + tileH * 2 + footerH;

    cv::namedWindow("Light Field Studio", cv::WINDOW_NORMAL);
    cv::resizeWindow("Light Field Studio", canvasW, canvasH);

    cv::createTrackbar("Focus", "Light Field Studio", &depthSlider, 120);
    cv::createTrackbar("Aperture", "Light Field Studio", &aperSlider, 15);

    // All-focus precompute
    auto stack = buildFocalStack(lf, 0.0f, 1.2f, 13);
    cv::Mat allFocus = allInFocusFromStack(stack);

    while (true) {
        float d = depthSlider / 100.0f;
        float r = std::min((float)aperSlider,
                           (lf.lensletSize - 1) * 0.5f);

        cv::Mat ref = refocusShiftSum(lf, d);
        cv::Mat ap  = apertureAverage(lf, r);

        // Prepare canvas
        cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(30,30,30));

        // Header
        cv::putText(canvas, "Light Field Studio",
            {canvasW/2 - 120, 30},
            cv::FONT_HERSHEY_SIMPLEX, 1.0,
            {255,255,255}, 2);

        // Resize & convert
        cv::Mat m0, m1, m2, m3;
        cv::resize(toDisplay8U(mosaicLF), m0, {tileW, tileH});
        cv::resize(toDisplay8U(ref),       m1, {tileW, tileH});
        cv::resize(toDisplay8U(ap),        m2, {tileW, tileH});
        cv::resize(toDisplay8U(allFocus), m3, {tileW, tileH});

        // Place tiles
        m0.copyTo(canvas(cv::Rect(0, headerH, tileW, tileH)));
        m1.copyTo(canvas(cv::Rect(tileW, headerH, tileW, tileH)));
        m2.copyTo(canvas(cv::Rect(0, headerH + tileH, tileW, tileH)));
        m3.copyTo(canvas(cv::Rect(tileW, headerH + tileH, tileW, tileH)));

        // Labels
        drawTitle(canvas(cv::Rect(0, headerH, tileW, tileH)), "Mosaic");
        drawTitle(canvas(cv::Rect(tileW, headerH, tileW, tileH)), "Refocus");
        drawTitle(canvas(cv::Rect(0, headerH + tileH, tileW, tileH)), "Aperture");
        drawTitle(canvas(cv::Rect(tileW, headerH + tileH, tileW, tileH)), "All Focus");

        cv::imshow("Light Field Studio", canvas);

        int k = cv::waitKey(15);
        if (k == 27 || k == 'q') break;
    }

    return 0;
}
