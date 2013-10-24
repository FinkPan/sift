#ifndef _SIFT_HPP_
#define _SIFT_HPP_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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
    sift(){}
    sift(double raw, double col)
    {
        descriptor_.x = raw;
        descriptor_.y = col;
        descriptor_.scale = 0;
        descriptor_.orientation = 0;
    }
    //读取图片
    void LoadImage(const string &file_path);
    void AssignOrientations();                  //方向分配
    void DescriptorRepresentation();            //生成sift向量
    void write_features(string &file);

private:

    Descriptor descriptor_;
    Mat grayimage_;


private:
    //计算直方图
    void CalculateOrientationHistogram(const Mat& gauss, 
                                             int x, int y,          //图像坐标x为col, y为row
                                             int bins,              //柱数 为36
                                             int radius,            //radius = 3 * 1.5 = 4
                                             double sigma,          //加权 1.5*2 = 3
                                             vector<double>& hist); //每10度一个柱,将360度分为36柱

    //计算模值和方向：计算在gauss图像中坐标为xy处的模值和方向
    bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori);

    void NormalizeDescr(Descriptor& feat);//归一化处理

    double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width);
    void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d);
    //高斯平滑，模板为{0.25, 0.5, 0.25}
    void GaussSmoothOriHist(vector<double>& hist, int n);
    //4.3 直方图的极值查找
    double sift::DominantDirection(vector<double>& hist, int n);

};



#endif