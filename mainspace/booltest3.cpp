#include<iostream>
using namespace std;

void show(int***visarr,int rows,int cols ){

for(int i=0;i<rows;i++){

  for(int j=0;j<cols;j++){

  cout<<*(visarr[i][j])<<" ";

  }
cout<<endl;


}
}


int main(){
int a=10;
int rows=4,cols=3;
int *(**visarr)=new int**[rows];

for(int i=0;i<rows;i++){

visarr[i]=new int*[cols];
  for(int j=0;j<cols;j++){
    visarr[i][j]=&a;
  }

}




show(visarr,rows,cols);




/*
int* __restrict visarr[rows][cols]={NULL};


for(int i=0;i<rows;i++){

//visarr[i]=new int*[cols];
  for(int j=0;j<cols;j++){
   // int *p;
   // p=&visarr[i][j];
    //*p=a;      
    visarr[i][j]=&a;
  }

}



*/
/*
for(int i=0;i<rows;i++){

  for(int j=0;j<cols;j++){

  cout<<*(visarr[i][j])<<" ";

  }
cout<<endl;

}
*/

a=3;

show(visarr,rows,cols);



/*
for(int i=0;i<rows;i++){

  for(int j=0;j<cols;j++){

  cout<<*(visarr[i][j])<<" ";

  }
cout<<endl;

}

*/
}
