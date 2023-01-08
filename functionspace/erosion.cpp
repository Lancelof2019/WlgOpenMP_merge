#include "../headerspace/WatershedAlg.h"
using namespace cv;

Mat WatershedAlg::erosion(Mat &image, vector< vector <int> > &kernel) {


    int morph_size = 2;
    Mat element = getStructuringElement(cv::MORPH_RECT, Size(2 * morph_size + 1,2 * morph_size + 1),
    Point(morph_size, morph_size));
    
    Mat dill(image.rows,image.cols,CV_8UC1,Scalar::all(0));

    dilate(image, dill, element,Point(-1, -1), 1);

    int n = kernel.size();
    int m = kernel[0].size();
    image.release();
    vector< vector < bool> > shouldBeZeroImage(dill.rows, vector<bool>(dill.cols, false));
      // #pragma omp parallel for collapse(2) simd nowait
       #pragma omp parallel for  //simd 
       for(int i = n / 2; i < dill.rows - n / 2; i++) {  // pragma omp parallel for should be in front of a large loop!
       // #pragma omp parallel for //simd     
            for(int j = m / 2; j < dill.cols - m / 2; j++) {

                // Loop the kernel
                if( (int) dill.at<uchar>(i, j) == ONE) {
                    bool shouldBeZero = false;
                    int x=0;
                    //
                    //for(int crtX = i - n / 2,int x = 0; crtX <= i + n / 2; crtX++,x++) {
                    //#pragma omp parallel for //simd //collapse(2)
                    for(int crtX = i - n / 2; crtX <= i + n / 2; crtX++) {
                       // #pragma omp parallel for
                        int y=0;
                       // for(int crtY = j - m / 2, int y = 0; crtY <= j + m / 2; crtY++, y++) {
                      // #pragma omp parallel for 
                        for(int crtY = j - m / 2; crtY <= j + m / 2; crtY++) {
                            if((int) dill.at<uchar>(crtX, crtY) == ZERO && kernel[x][y] == 1) {
                                shouldBeZero = true;
                                break;//there is break,no use for openmp 

                                /*
                                ../functionspace/erosion.cpp:39:33: error: break statement used with OpenMP for loop
                                 break;
                                */
                            }
                          y++;
                            
                        }
                        
                        x++;

                    }

                    if(shouldBeZero) {
                        shouldBeZeroImage[i][j] = true;
                    }
                }
            }
        }

        #pragma omp parallel for 
	for(int i = 0; i < dill.rows; i++) {
           //#pragma omp parallel for 
            for(int j = 0; j < dill.cols; j++) {
                if(shouldBeZeroImage[i][j]) {
                    dill.at<uchar>(i, j) = ZERO;
                }
            }
        }

    return dill;

    }
