#include "lightfield.hpp"
#include <cmath>
#include <algorithm>

static inline cv::Mat toFloat01(const cv::Mat& img) {
    cv::Mat out;
    if (img.depth() == CV_8U) {
        img.convertTo(out, CV_32F, 1.0 / 255.0);
    } else if (img.depth() == CV_16U) {
        img.convertTo(out, CV_32F, 1.0 / 65535.0);
    } else if (img.depth() == CV_32F) {
        out = img.clone();
    } else {
        cv::Mat tmp;
        img.convertTo(tmp, CV_32F);
        out = tmp;
    }
    return out;
}

void LightField::buildFromMosaic(const cv::Mat& mosaic) {
    ready = false;
    if (mosaic.empty()) return;

    cv::Mat m = mosaic;
    if (m.channels() == 1) cv::cvtColor(m, m, cv::COLOR_GRAY2BGR);
    if (m.channels() == 4) cv::cvtColor(m, m, cv::COLOR_BGRA2BGR);

    m = toFloat01(m);

    U = lensletSize;
    V = lensletSize;

    // Typical plenoptic mosaic: pixels are interleaved by angular indices
    // subView(v,u) = m(v::V, u::U)
    S = m.rows / V;
    T = m.cols / U;
    if (S <= 0 || T <= 0) return;

    subViews.assign(V, std::vector<cv::Mat>(U));

    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
            cv::Mat sv(S, T, CV_32FC3);
            for (int s = 0; s < S; ++s) {
                const cv::Vec3f* srcRow = m.ptr<cv::Vec3f>(v + s * V);
                cv::Vec3f* dstRow = sv.ptr<cv::Vec3f>(s);
                for (int t = 0; t < T; ++t) {
                    dstRow[t] = srcRow[u + t * U];
                }
            }
            subViews[v][u] = sv;
        }
    }

    ready = true;
}

static inline cv::Mat shiftImage(const cv::Mat& img, float dx, float dy) {
    cv::Mat out;
    cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(img, out, M, img.size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT101);
    return out;
}

// Report-style diagonal shifting setup (centered coordinates).
// We use centered u,v offsets: uOff = (maxUV - u), vOff = (v - maxUV)
// which matches the "u decreases while v increases" diagonal idea. :contentReference[oaicite:2]{index=2}
cv::Mat refocusShiftSum(const LightField& lf, float d) {
    CV_Assert(lf.ready);

    const float maxUV = (lf.lensletSize - 1) * 0.5f;
    cv::Mat acc = cv::Mat::zeros(lf.S, lf.T, CV_32FC3);

    for (int v = 0; v < lf.V; ++v) {
        for (int u = 0; u < lf.U; ++u) {
            float uOff = ((float)u-maxUV);
            float vOff = (maxUV-(float)v);

            // d controls how much shift we apply (synthetic refocus) :contentReference[oaicite:3]{index=3}
            cv::Mat shifted = shiftImage(lf.subViews[v][u], d * uOff, d * vOff);
            acc += shifted;
        }
    }
    acc *= (1.0f / (lf.U * lf.V));
    return acc;
}

cv::Mat apertureAverage(const LightField& lf, float radius) {
    CV_Assert(lf.ready);

    const float maxUV = (lf.lensletSize - 1) * 0.5f;
    cv::Mat acc = cv::Mat::zeros(lf.S, lf.T, CV_32FC3);
    int count = 0;

    for (int v = 0; v < lf.V; ++v) {
        for (int u = 0; u < lf.U; ++u) {
            float du = (float)u - maxUV;
            float dv = (float)v - maxUV;
            float r = std::sqrt(du*du + dv*dv);
            if (r <= radius + 1e-6f) {
                acc += lf.subViews[v][u];
                count++;
            }
        }
    }
    if (count > 0) acc *= (1.0f / count);
    return acc;
}

cv::Mat makeSubApertureMosaic(const LightField& lf, int tileW) {
    CV_Assert(lf.ready);

    int tileH = (int)std::round(tileW * (lf.S / (float)lf.T));
    cv::Mat canvas(tileH * lf.V, tileW * lf.U, CV_32FC3, cv::Scalar(0,0,0));

    for (int v = 0; v < lf.V; ++v) {
        for (int u = 0; u < lf.U; ++u) {
            cv::Mat resized;
            cv::resize(lf.subViews[v][u], resized, cv::Size(tileW, tileH), 0, 0, cv::INTER_AREA);
            resized.copyTo(canvas(cv::Rect(u * tileW, v * tileH, tileW, tileH)));
        }
    }
    return canvas;
}

std::vector<cv::Mat> buildFocalStack(const LightField& lf,
                                     float dMin, float dMax, int steps) {
    CV_Assert(lf.ready);
    steps = std::max(2, steps);
    std::vector<cv::Mat> stack;
    stack.reserve(steps);

    for (int i = 0; i < steps; ++i) {
        float a = (float)i / (steps - 1);
        float d = dMin * (1.0f - a) + dMax * a;
        stack.push_back(refocusShiftSum(lf, d));
    }
    return stack;
}

static inline cv::Mat luminance(const cv::Mat& rgb01) {
    // rgb01 is CV_32FC3 in [0,1]
    std::vector<cv::Mat> ch;
    cv::split(rgb01, ch);
    return 0.2126f * ch[2] + 0.7152f * ch[1] + 0.0722f * ch[0]; // assuming BGR ordering -> ch[2]=R
}

// Implements the sharpness-weighted fusion equations in your report. :contentReference[oaicite:4]{index=4}
cv::Mat allInFocusFromStack(const std::vector<cv::Mat>& focalStack,
                            float sigmaLow, float sigmaSharp) {
    CV_Assert(!focalStack.empty());

    const int H = focalStack[0].rows;
    const int W = focalStack[0].cols;

    cv::Mat num = cv::Mat::zeros(H, W, CV_32FC3);
    cv::Mat den = cv::Mat::zeros(H, W, CV_32F);

    for (const auto& I : focalStack) {
        cv::Mat Ilum = luminance(I);

        cv::Mat Ilow, Ihigh;
        cv::GaussianBlur(Ilum, Ilow, cv::Size(0,0), sigmaLow);
        Ihigh = Ilum - Ilow;

        cv::Mat w = Ihigh.mul(Ihigh);
        cv::GaussianBlur(w, w, cv::Size(0,0), sigmaSharp);

        // Accumulate
        std::vector<cv::Mat> ch;
        cv::split(I, ch);
        for (int c = 0; c < 3; ++c) ch[c] = ch[c].mul(w);
        cv::Mat weighted;
        cv::merge(ch, weighted);

        num += weighted;
        den += w;
    }

    cv::Mat out;
    std::vector<cv::Mat> outCh(3);
    cv::Mat denSafe = den + 1e-8f;
    cv::split(num, outCh);
    for (int c = 0; c < 3; ++c) outCh[c] = outCh[c] / denSafe;
    cv::merge(outCh, out);

    return out;
}

cv::Mat depthFromRefocus(const std::vector<cv::Mat>& focalStack,
                         float dMin, float dMax) {
    CV_Assert(!focalStack.empty());
    int steps = (int)focalStack.size();

    cv::Mat num = cv::Mat::zeros(focalStack[0].rows, focalStack[0].cols, CV_32F);
    cv::Mat den = cv::Mat::zeros(focalStack[0].rows, focalStack[0].cols, CV_32F);

    for (int i = 0; i < steps; ++i) {
        float a = (float)i / (steps - 1);
        float d = dMin * (1.0f - a) + dMax * a;

        cv::Mat Ilum = luminance(focalStack[i]);
        cv::Mat Ilow, Ihigh;
        cv::GaussianBlur(Ilum, Ilow, cv::Size(0,0), 1.0);
        Ihigh = Ilum - Ilow;

        cv::Mat w = Ihigh.mul(Ihigh);
        cv::GaussianBlur(w, w, cv::Size(0,0), 2.0);

        num += w * d;
        den += w;
    }

    cv::Mat depth = num / (den + 1e-8f);

    // normalize 0..1 for display
    double mn, mx;
    cv::minMaxLoc(depth, &mn, &mx);
    cv::Mat out = (depth - (float)mn) / (float)(mx - mn + 1e-8);
    return out;
}
