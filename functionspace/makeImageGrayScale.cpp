#include "../headerspace/WatershedAlg.h"
using namespace cv;

Mat WatershedAlg::makeImageGrayScale(Mat &image) {
      
      Mat grayScale(image.rows, image.cols, CV_8UC1, Scalar::all(0));
  
        double gray=0.0,gray1=0.0;
     
      #pragma omp parallel for  //parallel for simd  
      for(int i = 0; i < image.rows; i++) {
          #pragma omp parallel for  
            for(int j = 0; j < image.cols; j++) {
                 gray = 0.21 * image.at<cv::Vec3b>(i,j)[0] +
                              0.72 * image.at<cv::Vec3b>(i,j)[1] +
                              0.07 * image.at<cv::Vec3b>(i,j)[2];

                grayScale.at<uchar>(i,j) = (uchar)gray;
            }
        }
 
   
       
        cv::GaussianBlur(grayScale, grayScale, Size(3,5), 3,4);
       
        image.release();
        return grayScale;
    }

