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
    //������λ��
    double x;  //ͼ��x����
    double y;  //ͼ��y����

    //�����ӳ߶�
    double scale;

    //�����ӷ���
    double orientation;

    //128ά��������
    double descriptorvector[128]; //Ĭ��Ϊ128ά����

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
    //��ȡͼƬ
    void LoadImage(const string &file_path);
    void AssignOrientations();                  //�������
    void DescriptorRepresentation();            //����sift����
    void write_features(string &file);

private:

    Descriptor descriptor_;
    Mat grayimage_;


private:
    //����ֱ��ͼ
    void CalculateOrientationHistogram(const Mat& gauss, 
                                             int x, int y,          //ͼ������xΪcol, yΪrow
                                             int bins,              //���� Ϊ36
                                             int radius,            //radius = 3 * 1.5 = 4
                                             double sigma,          //��Ȩ 1.5*2 = 3
                                             vector<double>& hist); //ÿ10��һ����,��360�ȷ�Ϊ36��

    //����ģֵ�ͷ��򣺼�����gaussͼ��������Ϊxy����ģֵ�ͷ���
    bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori);

    void NormalizeDescr(Descriptor& feat);//��һ������

    double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width);
    void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d);
    //��˹ƽ����ģ��Ϊ{0.25, 0.5, 0.25}
    void GaussSmoothOriHist(vector<double>& hist, int n);
    //4.3 ֱ��ͼ�ļ�ֵ����
    double sift::DominantDirection(vector<double>& hist, int n);

};



#endif