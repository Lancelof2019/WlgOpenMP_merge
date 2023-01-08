#include "../headerspace/WatershedAlg.h"
#include <cmath> 
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include"btfsFunc.cpp"
using namespace cv;

Array2D<int> &WatershedAlg::thresholdMeasure(Mat &image,Array2D<int> &threshmat) {

 cv::adaptiveThreshold(image,image,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,  29 ,4);

 #pragma omp parallel for
 for(int i=0;i<image.rows;i++){
     #pragma omp parallel for
	 for(int j=0;j<image.cols;j++){
         threshmat(i,j)=(int)image.at<uchar>(i,j);

	}
}
/*
int maxdist=0;
#pragma omp parallel for  reduction(max:maxdist)
for(int i=0;i<image.rows;i++){
  for(int j=0;j<image.cols;j++){
   // bool **vist=new bool*[rows]; 
     //  for(int m=0;m<rows;m++){
       //   vist[i]=new bool[cols];
         // for(int n=0;n<cols;n++){
        //      vist[m][n]=false;
        // }
     // }
   if(threshmat(i,j)!=0){
   // if(platmarker[i][j]==1){
      bfs(i,j,threshmat,distance,image.rows,image.cols);
       if(maxdist<=distance[i][j]){
          maxdist=distance[i][j];
     }
   // }
  }
 }
}

//cout<<"maxdist is :"<<maxdist<<endl;



for(int i=0;i<image.rows;i++){
  for(int j=0;j<image.cols;j++){
  if(distance[i][j]==0){
 threshmat(i,j)=threshmat(i,j)*maxdist+distance[i][j];
  }
  else
    threshmat(i,j)=threshmat(i,j)*maxdist+distance[i][j]-1;
  }
}
       

*/

        return threshmat;
    }
