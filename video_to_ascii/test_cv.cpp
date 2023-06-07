/*
g++ -std=c++11 test_cv.cpp -o test_cv `pkg-config --cflags --libs opencv4`
*/


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("path_to_image.png");

    if (image.empty()) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    imshow("Image", image);
    waitKey(0);

    return 0;
}
