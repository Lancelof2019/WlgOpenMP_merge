#include "../headerspace/WatershedAlg.h"
#include <queue>
#include <cstdlib> 
#include <vector>
#include <string.h>
#include<iostream>
#include "btfsFunc.cpp"
#define NUMSIZE 8
#define NSIZE 4

using namespace cv;

Array2D<int> &WatershedAlg::distanceTransform(Array2D<int>& matArr, Array2D<int> &markers,int &rows,int &cols,Array2D<int> &plots,Array2D<bool>& visArr,Array1D &plotx,Array1D &ploty,int**** arr4D,int**** mat4D,Bool2D &visBool,int **platmarker,int **distance) {

       // queue<int> qx, qy;
        int dx[NUMSIZE]={-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[NUMSIZE]={0, 0, -1, 1, -1,  1, 1, -1};
        int pixelThreshold=1;
int boundcounter=0;
int zerocounter=0;
#pragma omp parallel for reduction(+:boundcounter,zerocounter)	   
for(int i=0;i<rows;i++){
   for(int j=0;j<cols;j++){
     if(matArr(i,j)==ZERO){
  
             visArr(i,j)=true; 
            // zerocounter+=1;
	     continue;
            }

        for(int k=0;k<NSIZE;k++){
	   // if(!visArr(arr4D[i][j][0][k],arr4D[i][j][1][k])){
            if(arr4D[i][j][0][k]>-1 && arr4D[i][j][1][k]>-1 && arr4D[i][j][0][k]<rows && arr4D[i][j][1][k]<cols){
                
	       if(!visArr(arr4D[i][j][0][k],arr4D[i][j][1][k])) {  
            
                    if(matArr(arr4D[i][j][0][k],arr4D[i][j][1][k])==ZERO){ //!visArr(arr4D[i][j][0][k],arr4D[i][j][0][k])){  
                       matArr(i,j)=pixelThreshold;
                       visArr(i,j)=true;
		      // boundcounter+=1;
		       plots(i,j)=1;

                     //  mat4D[i][j][0][k]=i;
                     //  mat4D[i][j][1][k]=j;
                      // vcounter++;
                 }
	       }
	    }
             // }
	 }
       }
    }
   int maxVal=0;
        int pcounter=0;
//        #pragma omp parallel for reduction(+:pcounter)
// this is very inefficient on GPU (reduction kernel!) and serial on CPU !!!
	for(int i=0;i<rows;i++){
  //         #pragma omp parallel for
           for(int j=0;j<cols;j++){
             if(plots(i,j)==1){
                  plotx(pcounter)=i;
		  ploty(pcounter)=j;
                 // qx.push(i);
		 // qy.push(j);
		  pcounter++;
	     }
              

	   }


	}

int qcounter=0;
int i=0;
while(plotx(i)!=-1){

            int crtX=plotx(i);
            int crtY=ploty(i);
             i++;
             qcounter++;
            bool isBigger = true;//check
          // #pragma omp simd
            for(int h = 0; h < NUMSIZE; h++) {
                int nextX = crtX + dx[h];
                int nextY = crtY + dy[h];

                if( nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols || matArr(nextX,nextY) == ZERO ) {
                    continue;
                }

                if( matArr(crtX,crtY) <= matArr(nextX,nextY)) {
                    isBigger = false;

                }
                //pick the max local value of some regions

                if( matArr(crtX,crtY) +1< matArr(nextX,nextY)) {
                    visArr(nextX,nextY) = true;
                    matArr(nextX,nextY) =  min((matArr(crtX,crtY)+1), 254);
              //    matArr(nextX,nextY) =  matArr(crtX,crtY)+1;
                    //to get max value for difference between max value image and image
                    if(maxVal<=matArr(nextX,nextY)){
                       maxVal=matArr(nextX,nextY);

                    }
                     //to get max value for difference between max value image and image
 
                    plotx(pcounter)=nextX;
                    ploty(pcounter)=nextY;
                    pcounter++;

                }
                //fout4<<endl;
            }
           //find the max value in local area
            if(isBigger) {
                markers(crtX,crtY)=2;
             }
          }

   pixelThreshold=pixelThreshold-1;
   int pnumThrshold=2000;
   int handlingType=0;
   int neighbourType=0;
   removeholesopt(matArr,pnumThrshold,  handlingType, neighbourType, pixelThreshold,rows,cols);
/*
#pragma omp parallel for
 for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++){
    int sameNeigh=0;
     for(int h=0;h<4;h++){
     int nextX=i+dx[h];
     int nextY=j+dy[h];
     if( nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols || matArr(nextX,nextY) == ZERO) {
         continue;
        }
      if(matArr(i,j)==matArr(nextX,nextY)){
           sameNeigh++;
           
	   }
      

    }
    if(sameNeigh==4){
        platmarker[i][j]=1;
    }
   
  }

 }

  */ 

/*
for(int i=0;i<rows;i++){
 for(int j=0;j<cols;j++){
  if(matArr(i,j)!=0){
   


  }	 
 }
}
*/


/*
int maxdist=0;
#pragma omp parallel for  reduction(max:maxdist)
for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++){
   // bool **vist=new bool*[rows]; 
     //  for(int m=0;m<rows;m++){
       //   vist[i]=new bool[cols];
         // for(int n=0;n<cols;n++){
	//	vist[m][n]=false;
        // }
     // }
   if(matArr(i,j)!=0){	  
   // if(platmarker[i][j]==1){
      bfs(i,j,matArr,distance,rows,cols);
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
  matArr(i,j)=matArr(i,j)*maxdist+distance[i][j];
  }
  else
    matArr(i,j)=matArr(i,j)*maxdist+distance[i][j]-1;
  }
}

*/



     /*   string filename25="/opt/test25.txt";
        ofstream fout25(filename25);
        



        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {

             fout25<<((int)matArr(i,j))<<",";
             

        }
         fout25<<endl;
     }
     fout25.close();
*/
      return matArr;
    }
