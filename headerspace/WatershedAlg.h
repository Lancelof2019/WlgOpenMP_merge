#ifndef WATERSHEDALG_H
#define WATERSHEDALG_H
#include"../functionspace/buffer.cpp"
//#include "../structPixel/structPixel.cpp"
#include<iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include<queue>
#include <vector>
#include<random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <queue>
#include<string>
#include <cstdlib>
#include<omp.h>
using namespace cv;
using namespace std;
#define ZERO 0
#define ONE 254
#define STEP 1
//#include <opencv2/imgcodecs.hpp>
class WatershedAlg{

public:
Mat makeImageGrayScale(Mat &image);
Mat inverseImage(Mat &image);
Mat erosion(Mat &image, vector< vector <int> > &kernel);
Array2D<int> &thresholdMeasure(Mat &image,Array2D<int> &thrshmat0);
Array2D<int> &antiInverseImage(Array2D<int> &threshmat,int &rows,int &cols);
Array2D<int> &distanceTransform(Array2D<int>&matArr, Array2D<int> &markers,int &rows,int &cols,Array2D<int>& plots,Array2D<bool> &visArr,Array1D &plotx,Array1D& ploty,int ****arr4D,int**** mat4D,Bool2D &visBool,int **platmarker,int **distance); 
Mat watershed(Array2D<int> &image, Array2D<int> &markers,Mat &duplImage,int &rows,int &cols,Array2D<bool> &inprioq,Array2D<int> &markerMap,Array2D<int> &temp,Array2D<int> &nextSet,int& id_num);
void removeholesopt(Array2D<int> &src, int &pnumlThrshold, int &handlingType, int &neighbourType,int pixelThreshhold,int &rows,int &cols);
void processImage(Mat &image,Mat &duplImage,Array2D<int> &threshmat,Array2D<int> &markers,Array2D<bool> &visArr,Array1D &plotx,Array1D &ploty,Array2D<int> &plots,Array2D<bool> &inprioq,Array2D<int> &markerMap,Array2D<int> &temp,Array2D<int> &nextSet,int**** arr4D,int**** mat4D,Bool2D &visBool,int **platmarker,int **distance) ;
};
#endif
