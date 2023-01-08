#include "../headerspace/WatershedAlg.h"
#include<opencv2/ximgproc.hpp>
using namespace cv;
using namespace ximgproc;
void WatershedAlg::processImage(Mat &image,Mat &duplImage,Array2D<int> &threshmat0,Array2D<int> & markers,Array2D<bool> &visArr,Array1D &plotx,Array1D &ploty,Array2D<int>& plots,Array2D<bool> &inprioq,Array2D<int> &markerMap,Array2D<int> &temp,Array2D<int>& nextSet,int**** arr4D,int**** mat4D,Bool2D &visBool,int **platmarker,int **distance) {
       //Mat testImg;
//       image.copyTo(testImg);
       image = makeImageGrayScale(image);


       vector< vector<int> > erosionTargetThree{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, };
       image = erosion(image, erosionTargetThree);

       // int** __restrict threshmat;
       
        threshmat0= thresholdMeasure(image,threshmat0);

       //int** __restrict antimat;
       threshmat0=antiInverseImage(threshmat0,image.rows,image.cols);

	//int** __restrict matimage;
       threshmat0=distanceTransform(threshmat0,markers,image.rows,image.cols,plots,visArr,plotx,ploty,arr4D,mat4D,visBool,platmarker,distance);

       int id_num=0;
       image = watershed(threshmat0, markers,duplImage,image.rows,image.cols,inprioq,markerMap,temp,nextSet,id_num);
      // cv::imshow("watershed result",image);
      // cv::waitKey(0);
       // id_num=40;   
       //int avgPixels=(image.rows*image.cols)/id_num;
//        Mat matlabel;
//	Mat matmask;
       // Mat testlabel;
       //Mat testmask;
//       Ptr<SuperpixelLSC> SlicImg=createSuperpixelLSC(image,150,0.05);
      // Ptr<SuperpixelLSC> supertestImg=createSuperpixelLSC(testImg,80,0.07);

      //supertestImg->iterate(15);
      //supertestImg->getLabels(testlabel);
      //supertestImg->getLabelContourMask(testmask);

//       SlicImg->iterate(20);
//       SlicImg->getLabels(matlabel);
//       SlicImg->getLabelContourMask(matmask);
       //SlicImg=cv::ximgproc::createSuperpixelSLIC(image,SLIC,avgPixels,10);
       //
       //
      // testImg.setTo(Scalar(255,255,255),testmask);
  //     image.setTo(Scalar(255,255,255),matmask);

      // cv::imshow("merge testSLIC",testImg);
      // cv::waitKey(0);

    //   cv::imshow("merge SLIC",image);
    //   cv::waitKey(0);


    }
