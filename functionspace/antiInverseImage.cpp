
#include "../headerspace/WatershedAlg.h"
//#include"btfsFunc.cpp"
using namespace cv;

Array2D<int> &WatershedAlg::antiInverseImage(Array2D<int>&antimat,int &rows,int &cols) {
       #pragma omp parallel for 
	for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if( antimat(i,j) == ZERO ) {
                    antimat(i,j) = ZERO;
                } else {
                    antimat(i,j) = ONE;
                }
            }
        }


/*


int maxdist=0;
#pragma omp parallel for  reduction(max:maxdist)
for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++){
   if(antimat(i,j)!=0){
   // if(platmarker[i][j]==1){
      bfs(i,j,antimat,distance,rows,cols);
       if(maxdist<=distance[i][j]){
          maxdist=distance[i][j];
     }
   // }
  }
 }
}

//cout<<"maxdist is :"<<maxdist<<endl;



for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++){
  if(distance[i][j]==0){
 antimat(i,j)=antimat(i,j)*maxdist+distance[i][j];
  }
  else
    antimat(i,j)=antimat(i,j)*maxdist+distance[i][j]-1;
  }
}


*/




        return antimat;
    }
