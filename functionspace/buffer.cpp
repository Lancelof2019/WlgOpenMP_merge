#include<iostream>
#include <string.h>
#include<queue>
using namespace std;


/*
bool cmp1(int* para1,int* para2){

return para1[0]>para2[0];
}


struct Compare{

   bool operator()(int* para1, int* para2) {
        return para1[0] < para2[0];
    }

};

*/
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
  // memset(strarr,defaultVal,sizeof(int)*rows*cols);
}
bool* &operator()(int m,int n){
 return boolarr[m][n];
 }
//bool** operator()(const int&k){
// return matImg[k];
};


/*
int main(){
*/
/*
Array2D<int> mat=Array2D<int>(2,3,0);//Array2D<int>(2,3);

cout<<mat.matImg[0][1]<<endl;
cout<<mat(1,2)<<endl;
mat(1,2)=23;
mat(0,1)=mat(0,1)+3;
cout<<mat.arows<<mat.acols<<endl;

for(int i=0;i<mat.arows;i++){

 for(int j=0;j<mat.acols;j++){
   cout<<"mat("<<i<<","<<j<<") :"<<mat(i,j)<<endl;

 }


}

*/
/*
int rows=6;
int cols=10;
Array2D<int> mat1=Array2D<int>(rows,cols,false);//Array2D<int>(2,3);
#pragma acc parallel for
for(int i=0;i<mat1.arows;i++){

 for(int j=0;j<mat1.acols;j++){
  // cout<<"mat1("<<i<<","<<j<<") :"<<mat1(i,j)<<endl;



 }


}
*/
/*
Array1D arr(4,2,0);

cout<<arr(3)<<endl;

arr(2)=arr(2)+4;

int op=2;
int po=1;


cout<<arr(op)<<endl;

for(int i=0;i<arr.rows*arr.cols;i++){
cout<<"arr("<<i<<")"<<arr(i)<<endl;

}

Array2D<int> mat2(3,4,0);


mat2(2,3)=mat2(2,3)+2;

for(int i=0;i<mat2.arows;i++){

 for(int j=0;j<mat2.acols;j++){
   cout<<mat2(i,j)<<" ";

 }
cout<<endl;

}

mat2(0,0)=2;
mat2(1,0)=1;
mat2(2,0)=5;
//mat2(3,0)=3;


priority_queue<int*,vector<int*>,Compare> priq;

priq.push(mat2(1));
priq.push(mat2(2));
priq.push(mat2(0));
//priq.push(mat2(3));


for(int i=0;i<4;i++){

cout<<priq.top()[0]<<endl;
priq.pop();
}
*/
/*
}
*/
