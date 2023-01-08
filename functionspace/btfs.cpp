#include<iostream>
#include<queue>

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


void createVal(int arr[][16]){


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

void bfs(int x,int y,int arr[][16],bool visr[][16]){
queue<Pixel> q;
int dist=0;
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
cout<<"-------------------------------------------"<<endl;
cout<<"curx:"<<curX<<endl;
cout<<"curY:"<<curY<<endl;
cout<<"pvalue:"<<arr[curX][curY]<<endl;
cout<<"-------------------------------------------"<<endl;

for(int i=0;i<4;i++){
 int nextX=curX+dx[i];
 int nextY=curY+dy[i];

 if(nextX < 0 || nextY < 0 || nextX> 15 || nextY > 15 ||arr[nextX][nextY]>arr[curX][curY]) {
   continue;
 }

 if(!visr[nextX][nextY]&&(arr[nextX][nextY]<arr[curX][curY])){
    visr[nextX][nextY]=true;
    curp.value++;
    dist=curp.value;
    cout<<"The end of pixel inquery is:"<<"["<<nextX<<","<<nextY<<"]:"<<arr[nextX][nextY]<<endl;
    queue<Pixel>empq;
    swap(empq,q);
    break;
   }
   

  else if(arr[nextX][nextY]==arr[curX][curY]){
         //++allSame;
	 if(!visr[nextX][nextY]){
	 visr[nextX][nextY]=true;
	 int a=0;
         Pixel next=Pixel(nextX,nextY,a);
	 next.value=curp.value+1;
         cout<<"Pixel"<<"["<<nextX<<","<<nextY<<"]:"<<arr[nextX][nextY]<<endl;
         q.push(next);
	// if(allSame==4){
        // dist=dist+1;
	// cout<<"bist"<<dist<<endl;
	// if(minVal>=dist[i]){
         // minVal=dist[i];
	//   }
	  }
	 }
	

/*
  else if(!visr[nextX][nextY]&&arr[nextX][nextY]==arr[curX][curY]){
         visr[nextX][nextY]=true;
         Pixel next=Pixel(nextX,nextY,arr[nextX][nextY]);
         cout<<"Pixel"<<"["<<nextX<<","<<nextY<<"]:"<<arr[nextX][nextY]<<endl;
         q.push(next);
         dist[i]=dist[i]+1;
        }
*/

   }
 }

arr[x][y]=0;
cout<<"points"<<"["<<x<<","<<y<<"] :"<<temp<<endl;
//cout<<"dist is :"<<dist[0]<<endl;
//cout<<"dist is :"<<dist[1]<<endl;
//cout<<"dist is :"<<dist[2]<<endl;
//cout<<"dist is :"<<dist[3]<<endl;
//for(int i=0;i<4;i++){
//  if(minVal>=dist[i]){
//     minVal=dist[i];
 //  }
//
//}

cout<<"minVal is:"<<dist<<endl;
}

int main(){
int arr[16][16]={0};
bool visr[16][16]={false};
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
bfs(0,0,arr,visr);
cout<<endl;
for(int i=0;i<16;i++){
  for(int j=0;j<16;j++){
  cout<<arr[i][j];
  cout<<" ";

  }
  cout<<endl;
}


}
