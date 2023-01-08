#include<iostream>
#include<queue>
#include<omp.h>
using namespace std;
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

void bfs(int x,int y,int** arr,int **distance,int rows,int cols){
queue<Pixel> q;
int dist=0;
bool visr[rows][cols]={false};
Pixel p0=Pixel(x,y,dist);
q.push(p0);
visr[x][y]=true;
int temp;
temp=arr[x][y];
int dx[4]={-1, 1, 0, 0};
int dy[4]={0, 0, -1, 1};
while(!q.empty()){
Pixel curp=q.front();
q.pop();
int curX=curp.x;
int curY=curp.y;
for(int i=0;i<4;i++){
 int nextX=curX+dx[i];
 int nextY=curY+dy[i];
 if(nextX < 0 || nextY < 0 || nextX> 15 || nextY > 15 ||arr[nextX][nextY]>arr[curX][curY]) {
   continue;
 }

 if(!visr[nextX][nextY]&&(arr[nextX][nextY]<arr[curX][curY])){
    visr[nextX][nextY]=true;
    curp.value++;
    distance[x][y]=curp.value;
    queue<Pixel>empq;
    swap(empq,q);
    break;
   }
   

  else if(arr[nextX][nextY]==arr[curX][curY]){
	 if(!visr[nextX][nextY]){
	 visr[nextX][nextY]=true;
	 int a=0;
         Pixel next=Pixel(nextX,nextY,a);
	 next.value=curp.value+1;
         q.push(next);
	  }
	 }
   }
 }
}

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
