#include "MainWindow.hpp"
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>

static cv::Mat toDisplay8U(const cv::Mat& img01) {
    cv::Mat clipped, out;
    cv::min(cv::max(img01, 0.0f), 1.0f, clipped);
    clipped.convertTo(out, CV_8U, 255.0);
    return out;
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Light Field Studio");
    resize(1400, 900);

    QWidget* c = new QWidget(this);
    setCentralWidget(c);

    QVBoxLayout* main = new QVBoxLayout(c);
    QGridLayout* grid = new QGridLayout();

    mosaicLbl   = new QLabel("Mosaic");
    refocusLbl  = new QLabel("Refocus");
    apertureLbl = new QLabel("Aperture");
    allFocusLbl = new QLabel("All Focus");

    for (auto* l : {mosaicLbl, refocusLbl, apertureLbl, allFocusLbl}) {
        l->setMinimumSize(300, 220);
        l->setAlignment(Qt::AlignCenter);
        l->setStyleSheet("background:#222; color:white;");
    }

    grid->addWidget(mosaicLbl,   0, 0);
    grid->addWidget(refocusLbl,  0, 1);
    grid->addWidget(apertureLbl, 1, 0);
    grid->addWidget(allFocusLbl, 1, 1);

    main->addLayout(grid);

    QHBoxLayout* controls = new QHBoxLayout();
    focusSlider = new QSlider(Qt::Horizontal);
    aperSlider  = new QSlider(Qt::Horizontal);
    loadBtn     = new QPushButton("Load Image");

    focusSlider->setRange(0, 120);
    aperSlider->setRange(0, 15);

    controls->addWidget(new QLabel("Focus"));
    controls->addWidget(focusSlider);
    controls->addWidget(new QLabel("Aperture"));
    controls->addWidget(aperSlider);
    controls->addWidget(loadBtn);

    main->addLayout(controls);

    connect(focusSlider, &QSlider::valueChanged,
            this, [this](int v){ emit focusChanged(v / 100.0f); });

    connect(aperSlider, &QSlider::valueChanged,
            this, [this](int v){ emit apertureChanged((float)v); });

    connect(loadBtn, &QPushButton::clicked,
            this, &MainWindow::loadRequested);
}

void MainWindow::showMat(QLabel* lbl, const cv::Mat& img01) {
    if (img01.empty()) return;
    cv::Mat bgr = toDisplay8U(img01);
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage q(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    lbl->setPixmap(QPixmap::fromImage(q).scaled(
        lbl->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::setMosaic(const cv::Mat& img)   { showMat(mosaicLbl, img); }
void MainWindow::setRefocus(const cv::Mat& img)  { showMat(refocusLbl, img); }
void MainWindow::setAperture(const cv::Mat& img) { showMat(apertureLbl, img); }
void MainWindow::setAllFocus(const cv::Mat& img) { showMat(allFocusLbl, img); }
