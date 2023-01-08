 #include "../headerspace/WatershedAlg.h"
 #include<iostream>
// #include "../functionspace/reductionplat.cpp"
 #define NUMSIZE 8
 using namespace std;
 using namespace cv;
 #include <time.h>
 int main(){


cv::Mat srcImg=cv::imread("../images/istockphoto-1150481340-612x612.jpg");
//12_energy_map.png
//   cv::Mat srcImg=cv::imread("../image/pexels-sagui-andrea-618833.jpg");
   /*
	 Mat tarImg;

    srcImg.copyTo(tarImg);


    cv::Canny(srcImg,tarImg,200,230);
   cv::imshow("Canny edge",tarImg);
   cv::waitKey(0);
 */   
    //    cv::Mat srcImg=cv::imread("../images/les-nebo-oblaka-peizazh-gory-forest-sky-landscape-mountain-k.jpg");
//  cv::Mat srcImg=cv::imread("../images/matterhorn-1313x875.jpg");
//  cv::Mat srcImg=cv::imread("../images/istockphoto-1150481340-612x612.jpg");
     



//cv::Mat srcImg=cv::imread("../images/les-nebo-oblaka-peizazh-gory-forest-sky-landscape-mountain-k.jpg");




    Mat tarImg;

    srcImg.copyTo(tarImg);


    //cv::Canny(srcImg,tarImg,60,230);
   // cv::imshow("Canny edge",tarImg);
   // cv::waitKey(0);






     int rows=srcImg.rows;
     int cols=srcImg.cols;

     cv::Mat duplImage(srcImg.rows,srcImg.cols,CV_8UC3,Scalar::all(0));
     srcImg.copyTo(duplImage);

     if(!srcImg.data){

        cout<<"The file is not loaded or does not exist"<<endl;
        return -1;


     }

Array2D<int> threshmat(rows,cols,0);     
Array2D<int> markers(rows,cols,0);
Array2D<bool> visArr(rows,cols,false);
Array2D<int> plots(rows,cols,0);
Array2D<int> markerMap(rows,cols,0);
Array2D<bool> inprioq(rows,cols,false);
Array2D<int> temp(rows*cols,3,0);
Array2D<int> nextSet(rows*cols*9,3,0);
Bool2D visBool(rows,cols);

#pragma omp parallel for
for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++){
   visBool(i,j)=&visArr(i,j);
   
  }


}


int ** __restrict platmarker=new int*[rows];

for(int i=0;i<rows;i++){
   platmarker[i]=new int[cols];
   for(int j=0;j<cols;j++){
   platmarker[i][j]=0;
   }
}


int **__restrict distance=new int*[rows];
for(int i=0;i<rows;i++){
  // arr[i]=new int[cols];
   distance[i]=new int[cols];
   for(int j=0;j<cols;j++){
  //  arr[i][j]=0;
    distance[i][j]=0;
   }
}






int dx[NUMSIZE]={-1, 1, 0, 0, -1, -1, 1, 1};
int dy[NUMSIZE]={0, 0, -1, 1, -1,  1, 1, -1};

int nrows=2;
int ncols=8;
int ****arr4D=new int***[rows];
for(int i=0;i<rows;i++){
   arr4D[i]=new int**[cols];
     for(int j=0;j<cols;j++){
       arr4D[i][j]=new int*[nrows];
       for(int m=0;m<nrows;m++){
        arr4D[i][j][m]=new int[ncols];
	for(int n=0;n<ncols;n++){

	   arr4D[i][j][m][n]=0;

	}

       }
     }
}

#pragma omp parallel for
for(int i=0;i<rows;i++){

	for(int j=0;j<cols;j++){

          for(int z=0;z<8;z++){
           arr4D[i][j][0][z]=i+dx[z];
           arr4D[i][j][1][z]=j+dy[z];

	  }
	}
}


int ****mat4D=new int***[rows];
for(int i=0;i<rows;i++){
   mat4D[i]=new int**[cols];
     for(int j=0;j<cols;j++){
       mat4D[i][j]=new int*[nrows];
       for(int m=0;m<nrows;m++){
        mat4D[i][j][m]=new int[ncols];
        for(int n=0;n<ncols;n++){

           mat4D[i][j][m][n]=0;

        }

       }
     }
}






Array1D plotx(rows,cols,-1);
Array1D ploty(rows,cols,-1);







   clock_t start1,end1;
    double start,end;
    start1=clock();
    start=omp_get_wtime();
     WatershedAlg *walg=new WatershedAlg();
     walg->processImage(srcImg,duplImage,threshmat,markers,visArr,plotx,ploty,plots,inprioq,markerMap,temp,nextSet,arr4D,mat4D,visBool,platmarker,distance);
     end=omp_get_wtime();
     end1=clock();
     //cout<<"CLOCKS_PER_SEC :"<<CLOCKS_PER_SEC<<endl;
     cout<<"The total time the parallel  algorithm used is :"<<(double)(end - start)<<"s"<<endl;
     cout<<"The total time the algorithm used is :"<<(double)(end1 - start1)/CLOCKS_PER_SEC<<endl;
     delete walg;
     return 0;
   } 
