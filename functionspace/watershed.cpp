#include "../headerspace/WatershedAlg.h"
#include<vector>
#include<math.h>
#include<cmath>
#include <limits>
using namespace cv;
#define NUMSIZE 8
#define NSIZE 4




struct Pixel {
    int val;
    int x;
    int y;

    Pixel(int val, int x, int y) : val(val), x(x), y(y) {}

};


struct cmp1{

   bool operator()(Pixel &para1, Pixel &para2) {
        return para1.val < para2.val;
    }

};


struct nngNode{

int x;
int y;
int pixelval;
int pixelnum;
double ndist;
nngNode(int x, int y, int pixelval,int pixelnum,double ndist) : x(x), y(y), pixelval(pixelval),pixelnum(pixelnum),ndist(ndist) {}

};

struct cmp2{

   bool operator()(nngNode &para1, nngNode &para2) {
        return para1.ndist > para2.ndist;
    }

};


Mat WatershedAlg::watershed(Array2D<int> &image, Array2D<int>& markers,Mat &duplImage,int &rows,int &cols,Array2D<bool> &inprioq,Array2D<int>& markerMap,Array2D<int>&temp,Array2D<int>& nextSet, int &id_num){


       int msize=0;
  
       int dx[NUMSIZE]={-1, 1, 0, 0, -1, -1, 1, 1};
       int dy[NUMSIZE]={0, 0, -1, 1, -1,  1, 1, -1};

        // Put markers in priority queue
        int id = 1;
        Mat testDuplicate;
        duplImage.copyTo(testDuplicate);
        Mat dstImage(duplImage.rows,duplImage.cols,CV_8UC3,Scalar::all(0));


int tempcounter=0;

   for(int y=0;y<rows;y++){

      for(int z=0;z<cols;z++){
 
           if(markers(y,z)==2){    
            markerMap(y,z) = id;
//	    image(y,z)=image(y,z)+25;
           // msize++;
             for(int i = 0; i < 4; i++) {

                int newX =y + dx[i];
                int newY =z + dy[i];
                if(newX < 0 || newY < 0 || newX >= rows || newY >= cols) {
                    continue;
                }

	       temp(tempcounter,0)=image(newX,newY);
               temp(tempcounter,1)=newX;
               temp(tempcounter,2)=newY;
              // prioq.push(temp(tempcounter));
	       tempcounter++;

                
             }

             id++;
           }
        }
   }

         id_num=id;

         Array2D<int> idArr(id,id,0);

         int pixelSum[id]={0};
	 int pixelNum[id]={0};
    

         



         cv::Vec3b colors[id];
         #pragma omp parallel for
         for(int i = 0; i <= id; i++) {
           Vec3b vecColor;
           vecColor[0] = rand()%255+0;
           vecColor[1] = rand()%255+1;
           vecColor[2] = rand()%255+2;          
           colors[i]=vecColor;
        }


int ptcounter=0;
Vec3b boundColor;
boundColor[0] = 0;
boundColor[1] = 0;
boundColor[2] = 0;

#pragma omp parallel for reduction(+:pixelSum[:id],pixelNum[:id])
for(int i=0;i<tempcounter;i++){
   Pixel origin = Pixel( temp(i,0),temp(i,1),temp(i,2));
  //int originx=;
  //int originy=;
  //int originval=;
  priority_queue<Pixel,vector<Pixel>,cmp1>prioq;
  prioq.push(origin);
  while(!prioq.empty()){
    int CrtX=prioq.top().x;
    int CrtY=prioq.top().y;

    prioq.pop();

    bool canLabel = true;
    int neighboursLabel = 0;
    for(int i = 0; i < 8; i++) {
       int nextX = CrtX + dx[i];
       int nextY = CrtY + dy[i];
       if(nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols) {
         continue;
       }


        if(markerMap(nextX,nextY) != 0 && image(nextX,nextY)!=0){
                    if(neighboursLabel == 0) {
                        neighboursLabel = markerMap(nextX,nextY);//using id value,all strats from their closest neighbour
			//idArr(neighboursLabel,neighboursLabel)=neighboursLabel;
                    } else {//this part tells us that if there is two points at the boundary to see if they are in the same classification,
                    //two classification there is no merge
                        if(markerMap(nextX,nextY) != neighboursLabel ) {
                            canLabel = false;
			    duplImage.at<Vec3b>(CrtX,CrtY)=boundColor;
                            idArr(neighboursLabel-1,markerMap(nextX,nextY)-1)=1;

                        }
                    }
                } 
	 else {
                if(inprioq(nextX,nextY) == false) {
                        inprioq(nextX,nextY) = true;//aviod duplicate point is chosen,point does not exist in marker or background
                        Pixel next=Pixel(image(nextX,nextY),nextX,nextY);
			
			prioq.push(next);
                        //the most important expending step,//only the point whose pixel value is 0 in Markermap
                    }
                }
                

	    }//inner for

             if(canLabel&&image(CrtX,CrtY)!=0) {
                 markerMap(CrtX,CrtY) = neighboursLabel;//in this way it tells us that the points in same region share the same id or label
                // idArr[neighboursLabel-1][neighboursLabel-1][0][2]=idArr[neighboursLabel-1][neighboursLabel-1][0][2]+image(CrtX,CrtY);//sum of pixel value
		// idArr[neighboursLabel-1][neighboursLabel-1][0][1]=idArr[neighboursLabel-1][neighboursLabel-1][0][1]+1;//num
		 
		 pixelSum[neighboursLabel-1]=pixelSum[neighboursLabel-1]+image(CrtX,CrtY);
		 pixelNum[neighboursLabel-1]=pixelNum[neighboursLabel-1]+1;
                 duplImage.at<Vec3b>(CrtX,CrtY)=colors[ markerMap(CrtX,CrtY) ];
            }
    }//while
}//for



int ragNode=0;


priority_queue<nngNode,vector<nngNode>,cmp2>nngprioq;

Array2D<double> neighDist(id,id,0);
//#pragma omp parallel for reduction(+:ragNode) 
for(int i=0;i<id;i++){
   for(int j=0;j<id;j++){
    if(idArr(i,j)==1&&pixelNum[i]>0&&pixelNum[j]>0){
    ragNode++;
   // double powerDis=((pixelSum[i]/pixelNum[i]-pixelSum[j]/pixelNum[j])/(rows*cols))*((pixelSum[i]/pixelNum[i]-pixelSum[j]/pixelNum[j])/(rows*cols));
    double powerDis=pow((((double)pixelSum[i])/(double)pixelNum[i]-((double)pixelSum[j])/(double)pixelNum[j]),2);
    double mulelement=((double)pixelNum[i])*((double)pixelNum[j]);
    double divelement=(double)(pixelNum[i]+pixelNum[j]);
    neighDist(i,j)=((mulelement*powerDis)/divelement)+1.0;
    nngNode newnode=nngNode(i,j,pixelSum[i],pixelNum[i],neighDist(i,j));
    nngprioq.push(newnode);
       }
     }
}

int ngnum=0;
vector<nngNode>nngvec;
while(!nngprioq.empty()){
	 
        nngNode prenode=nngNode(nngprioq.top().x,nngprioq.top().y,nngprioq.top().pixelval,nngprioq.top().pixelnum,nngprioq.top().ndist);	    nngprioq.pop();


        // ncounter++;	 
	if(!nngprioq.empty()){
         if(prenode.ndist==nngprioq.top().ndist&&prenode.y==nngprioq.top().x&&prenode.x==nngprioq.top().y){
            nngvec.push_back(prenode);
	    nngvec.push_back(nngprioq.top());
     	 }else{

            //  vistNode(nngvec,nngprioq);
                 for(int h=0;h<nngvec.size();h++){
                    if(nngvec.at(h).x==nngprioq.top().x){
                        nngprioq.pop();
			break;
		    }
		    else{
		      	nngvec.push_back(nngprioq.top());    
			nngprioq.pop();
			break;
		    }

		 }
	   }
	}
}


//priority_queue<nngNode,vector<nngNode>,cmp2>nngpq;

       cv::addWeighted(duplImage,0.1,testDuplicate,0.9,0,dstImage);

    
        duplImage.release();
        testDuplicate.release();
        return dstImage;
    }
