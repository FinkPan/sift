#include <sift.hpp>

int main()
{
    Mat inputimage = imread("..//data//img1_500.jpg",0);
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift sift1;
    sift1.dosift(inputimage,Mat(),keypoints,descriptors);

    Mat outputimage;
    drawKeypoints(inputimage,keypoints,outputimage);

    namedWindow("keypoints",0);
    imshow("keypoints",outputimage);
    waitKey();


    return 0;
}