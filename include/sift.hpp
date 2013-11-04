#ifndef _SIFT_HPP_
#define _SIFT_HPP_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/******************************* Defs and macros *****************************/

// default number of sampled intervals per octave
static const int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
static const float SIFT_SIGMA = 1.6f;

// default threshold on keypoint contrast |D(x)|
static const float SIFT_CONTR_THR = 0.04f;

// default threshold on keypoint ratio of principle curvatures
static const float SIFT_CURV_THR = 10.f;

// double image size before pyramid construction?
static const bool SIFT_IMG_DBL = true;

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#if 0
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif


struct Descriptor
{
    //描述子位置
    double x;  //图像x坐标
    double y;  //图像y坐标

    //描述子尺度
    double scale;

    //描述子方向
    double orientation;

    //128维特征向量
    double descriptorvector[128]; //默认为128维向量

};

class sift
{
public:
    sift( int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6);

    //建立高斯金字塔
    void buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyr, int nOctaves ) const;

    //建立DoG差分金字塔
    void buildDoGPyramid( const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr ) const;

    // Detects features at extrema in DoG scale space.  Bad features are discarded
    // based on contrast and ratio of principal curvatures.
    void findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
        std::vector<KeyPoint>& keypoints ) const;

    void dosift(InputArray _image, InputArray _mask,
                std::vector<KeyPoint>& keypoints,
                OutputArray _descriptors,
                bool useProvidedKeypoints = false) const;

private:
    static inline void
        unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale);

    static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma );

    //计算指定像素的梯度方向直方图
    static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                      float sigma, float* hist, int n );

    // Interpolates a scale-space extremum's location and scale to subpixel
    // accuracy to form an image feature. Rejects features with low contrast.
    // Based on Section 4 of Lowe's paper.
    static bool adjustLocalExtrema( const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
        int& layer, int& r, int& c, int nOctaveLayers,
        float contrastThreshold, float edgeThreshold, float sigma );

    static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
        int d, int n, float* dst );

    static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
        Mat& descriptors, int nOctaveLayers, int firstOctave );

private:
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;

};

#endif