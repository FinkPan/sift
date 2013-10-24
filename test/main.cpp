#include <sift.hpp>

int main()
{
    string imagefile = "..//data//1-0.jpg";
    string featurefile = "..//data//1-0.txt";
    sift mysift(463.136,296.659);
    mysift.LoadImage(imagefile);
    mysift.AssignOrientations();
    mysift.DescriptorRepresentation();
    mysift.write_features(featurefile);
    return 0;
}