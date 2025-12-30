#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include <opencv2/opencv.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

    // Display sinks (replace cv::imshow)
    void setMosaic(const cv::Mat& img);
    void setRefocus(const cv::Mat& img);
    void setAperture(const cv::Mat& img);
    void setAllFocus(const cv::Mat& img);

signals:
    void focusChanged(float d);
    void apertureChanged(float r);
    void loadRequested();

private:
    QLabel *mosaicLbl, *refocusLbl, *apertureLbl, *allFocusLbl;
    QSlider *focusSlider, *aperSlider;
    QPushButton *loadBtn;

    void showMat(QLabel* lbl, const cv::Mat& img);
};
