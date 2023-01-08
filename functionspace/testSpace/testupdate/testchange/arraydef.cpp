#include<iostream>
#include <string.h>
#include<queue>
using namespace std;

bool cmp1(int* para1,int* para2){

return para1[0]>para2[0];
}


struct Compare{

   bool operator()(int* para1, int* para2) {
        return para1[0] < para2[0];
    }

};


template<class T1>
class Array2D{

public:
	int arows=0;
	int acols=0;
        T1**  __restrict matImg;
Array2D(int rows,int cols,T1 defaultVal):arows(rows),acols(cols){

matImg=new T1*[rows];
for(int i=0;i<rows;i++){
   matImg[i]=new T1[cols];
   memset(matImg[i],defaultVal,sizeof(T1)*cols);
   }
}

T1 &operator()(int m, const int n){

 return matImg[m][n];


 }


T1 *operator()(int k){

 return matImg[k];

}


~Array2D(){

for(int i=0;i<arows;i++){
    delete[] matImg[i];
   }
   delete [] matImg;

 }

};

class Array1D{

public:
int rows=0;
int cols=0;
int* __restrict strarr;
Array1D(int x,int y,int defaultVal):rows(x),cols(y){

   strarr=new int[rows*cols];
   memset(strarr,defaultVal,sizeof(int)*rows*cols);
}

int &operator()(int z){
 return strarr[z];

 }

~Array1D(){
delete[] strarr;

}

};

class Bool2D{
public:
int rows=0;
int cols=0;
bool*** __restrict boolarr;
Bool2D(int x,int y):rows(x),cols(y){
   boolarr=new bool**[rows];
   for(int i=0;i<rows;i++){
    boolarr[i]=new bool*[cols];
      for(int j=0;j<cols;j++){
	boolarr[i][j]=NULL;
      }

   }
}
bool* &operator()(int m,int n){
 return boolarr[m][n];
 }
};

