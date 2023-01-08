#include<iostream>
#include<queue>
#include<omp.h>
#include"btfsFuc.hpp"
//#include"creareVal.hpp"
using namespace std;
int main(){
int rows=16;
int cols=16;
int **arr=new int*[rows];
int **distance=new int*[rows];
for(int i=0;i<rows;i++){
   arr[i]=new int[cols];
   distance[i]=new int[cols];
   for(int j=0;j<cols;j++){
    arr[i][j]=0;
    distance[i][j]=0;
   }
}


//int distance[16][16]={0};

int x=0;
int Val=1;
createVal(arr);
for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  cout<<arr[i][j];
  cout<<" ";
  }
  cout<<endl;
}
cout<<endl;
int maxdist=0;
#pragma omp parallel for reduction(max:maxdist) 
for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  bfs(i,j,arr,distance,rows,cols);
 if(maxdist<=distance[i][j]){
 maxdist=distance[i][j];
 }
  }
}
cout<<"the max value is :"<<maxdist<<endl;
cout<<endl;
for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  if(distance[i][j]==0){
  arr[i][j]=arr[i][j]*maxdist+distance[i][j];
  }	  
  else
    arr[i][j]=arr[i][j]*maxdist+distance[i][j]-1;
  }
}


for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  cout<<arr[i][j];
  cout<<" ";
  }
  cout<<endl;
}
cout<<endl;
}
