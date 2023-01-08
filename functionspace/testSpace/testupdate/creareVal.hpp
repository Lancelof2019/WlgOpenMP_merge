#include<iostream>
#include<queue>
#include<omp.h>

/*
struct Pixel{
int x;
int y;
int value;
Pixel(int cordx,int cordy,int pvalue){

  this->x=cordx;
  this->y=cordy;
  this->value=pvalue;
}
};
*/
void createVal(int **arr){
for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  if(i==0||i==15){
     arr[i][j]=1;
   }
   else if(j==0||j==15){
    arr[i][j]=1;
   }
   
   if(i==1&&j!=0&&j!=15||i==14&&j!=15&&j!=0){
    arr[i][j]=2;
   }
   else if(j==1&&i!=0&&i!=15||j==14&&i!=15&&i!=0){
    arr[i][j]=2;
   }

  if(i==2&&j!=0&&j!=15&&j!=14&&j!=1||i==13&&j!=0&&j!=15&&j!=14&&j!=1){
   arr[i][j]=3;
  }
  else if(j==2&&i!=0&&i!=15&&i!=14&&i!=1||j==13&&i!=0&&i!=15&&i!=14&&i!=1){
    arr[i][j]=3;
   }

  if(i==3&&j!=0&&j!=15&&j!=14&&j!=1&&j!=2&&j!=13||i==12&&j!=0&&j!=15&&j!=14&&j!=1&&j!=2&&j!=13){
   arr[i][j]=4;
  }
  else if(j==3&&i!=0&&i!=15&&i!=14&&i!=1&&i!=2&&i!=13||j==12&&i!=0&&i!=15&&i!=14&&i!=1&&i!=2&&i!=13){
    arr[i][j]=4;
   }

  if(i==4&&j!=0&&j!=15&&j!=14&&j!=1&&j!=2&&j!=13&&j!=3&&j!=12||i==11&&j!=0&&j!=15&&j!=14&&j!=1&&j!=2&&j!=13&&j!=3&&j!=12){
   arr[i][j]=5;
  }
  else if(j==4&&i!=0&&i!=15&&i!=14&&i!=1&&i!=2&&i!=13&&i!=12&&i!=3||j==11&&i!=0&&i!=15&&i!=14&&i!=1&&i!=2&&i!=13&&i!=12&&i!=3){
    arr[i][j]=5;
   }

  }

 }

for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  if(arr[i][j]==0){
   arr[i][j]=6;
  }
  }
 }
}
