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

   /*    friend ostream &operator<<(ostream &os, const Pixel &pixel) {
        os << "val: " << pixel.val << " x: " << pixel.x << " y: " << pixel.y;
        return os;

    }
    */
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
/*nngNode(int x,int y,int pixelnngVal,int pixelnngNum,double nngDist){

 this->cordx=x;
 this->cordy=y;
 this->pixelnngVal=pixelnngVal;
 this->pixelnngNum=pixelnngNum;
 this->nngDist=nngDist;

}
*/
};

struct cmp2{

   bool operator()(nngNode &para1, nngNode &para2) {
        return para1.ndist > para2.ndist;
    }

};
/*
void vistNode(vector<nngNode>&nngvec,priority_queue<nngNode,vector<nngNode>,cmp2> &nngprioq){
             for(int i=0;i<nngvec.size();i++){
                   if(nngvec.at(i).x==nngprioq.top().x){
                      cout<<"------------------In vector-------------------------"<<endl;
	              cout<<"distanceValue:"<<nngprioq.top().ndist<<endl;
                      cout<<"x:"<<nngprioq.top().x<<endl;
                      cout<<"y:"<<nngprioq.top().y<<endl;
                      cout<<"sumPixel:"<<nngprioq.top().pixelval<<endl;
                      cout<<"numPixel:"<<nngprioq.top().pixelnum<<endl;
                      nngprioq.pop();
		   }
	     }

}
*/

Mat WatershedAlg::watershed(Array2D<int> &image, Array2D<int>& markers,Mat &duplImage,int &rows,int &cols,Array2D<bool> &inprioq,Array2D<int>& markerMap,Array2D<int>&temp,Array2D<int>& nextSet, int &id_num){



//priority_queue<int*,vector<int*>,Compare>prioq;


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
	 /////////////////////////////////////
         Array2D<int> idArr(id,id,0);
//	 Array2D<double> neighDis(id,id,0);
         int pixelSum[id]={0};
	 int pixelNum[id]={0};
       //  memset(pixelNum,-10,sizeof(pixelNum));

         



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
/*
//cout<<"OK for now"<<endl;
int ****idArr=new int***[id];
for(int i=0;i<id;i++){
   idArr[i]=new int**[id];
     for(int j=0;j<cols;j++){
       idArr[i][j]=new int*[1];
       for(int m=0;m<1;m++){
        idArr[i][j][m]=new int[4];
        for(int n=0;n<4;n++){
           idArr[i][j][m][0]=i*3+j;
           idArr[i][j][m][1]=0;//number of pixel in region
           idArr[i][j][m][2]=0;//sum of pixel value
           idArr[i][j][m][3]=0;//is there any edge



        }

       }
     }
}
*/

//int idSum=0;

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
	  
	  
    // int curpoint[3]={prioq.top()[0],prioq.top()[1],prioq.top()[2]};
    //int curpointx=temp(i,1);
    //int curpointx=temp(i,2);
    prioq.pop();
    //int CrtX=curpoint[1];
    //int CrtY=curpoint[2];
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
			    //idSum++;


                   //         Vec3b vecColor;
                     //       vecColor[0] = rand()%255+0;
                       //     vecColor[1] = rand()%255+1;
                         //   vecColor[2] = rand()%255+2;

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
/*
     string filename31="./test31.txt";
     ofstream fout31(filename31);


        for(int i = 0; i < id; i++) {
          for(int j = 0; j < id; j++) {

          fout31<<idArr(i,j)<<",";
             

        }
         fout31<<endl;
     }
       fout31.close();
       */

//cout<<"id 5 value:"<<pixelSum[5]<<endl;
//cout<<"id 5 num:"<<pixelNum[5]<<endl;

//cout<<"id 34 value:"<<pixelSum[34]<<endl;
//cout<<"id 34 num:"<<pixelNum[34]<<endl;
       
//cout<<"id 89 value:"<<pixelSum[89]<<endl;
//cout<<"id 89 num:"<<pixelNum[89]<<endl;
//cout<<"idSum:"<<idSum<<endl;
//Array2D<double>nng(idSum,idSum,0.0);
//
//
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
    /*
    cout<<"------------------------------------"<<endl;
    cout<<"powerDis:"<<powerDis<<endl;
    cout<<"mulelement:"<<mulelement<<endl;
    cout<<"divelement:"<<divelement<<endl;
    cout<<"("<<i<<","<<j<<")"<<endl;
    cout<<"psum"<<pixelSum[i]<<endl;
    cout<<"pnum"<<pixelNum[i]<<endl;
    cout<<"distance of"<<"("<<i<<","<<j<<"):"<<((mulelement*powerDis)/divelement)<<endl;
    */
    neighDist(i,j)=((mulelement*powerDis)/divelement)+1.0;
    nngNode newnode=nngNode(i,j,pixelSum[i],pixelNum[i],neighDist(i,j));
    nngprioq.push(newnode);
       }
     }
}
//cout<<"---------------------nng queue---------------------------"<<endl;
int ngnum=0;
vector<nngNode>nngvec;
//nngNode prenode=(nngprioq.top().x,nngprioq.top().y,nngprioq.top().pixelval,nngprioq.top().pixelnum,nngprioq.top().ndist);
while(!nngprioq.empty()){
	 
        nngNode prenode=nngNode(nngprioq.top().x,nngprioq.top().y,nngprioq.top().pixelval,nngprioq.top().pixelnum,nngprioq.top().ndist);	    nngprioq.pop();
	 //cout<<"--------------while priority queue----------------------"<<endl;
         //cout<<ncounter<<"st Element:"<<endl;
         //cout<<"distanceValue:"<<prenode.ndist<<endl;
         //cout<<"x:"<<prenode.x<<endl;
        // cout<<"y:"<<prenode.y<<endl;
        // cout<<"sumPixel:"<<prenode.pixelval<<endl;
        // cout<<"numPixel:"<<prenode.pixelnum<<endl;
	 /*if(prenode.ndist==nngprioq.top().ndist&&prenode.y==nngprioq.top().x&&prenode.x==nngprioq.top().y){
            cout<<"lable:"<<prenode.x<<" merge lable:"<<nngprioq.top().x<<" into :"<<"("<<prenode.x<<","<<
           
	 }
       //  ncounter++;
       //  
	 */
         
	 /*
	 if(nngvec.size()!=0){
	      for(int i=0;i<nngvec.size();i++){
                 if(nngvec.at(i).x==nngprioq.top().x){
                 nngprioq.pop();
	      } 

	  }
	 }
         */
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

//cout<<"----------------------------------"<<endl;



priority_queue<nngNode,vector<nngNode>,cmp2>nngpq;

/*
for(int i=0;i<nngvec.size();i++){
*/
/*
 * *
         cout<<"--------------------vector of nng nodes------------------------"<<endl;
         cout<<"distanceValue:"<<nngvec.at(i).ndist<<endl;
         cout<<"x:"<<nngvec.at(i).x<<endl;
         cout<<"y:"<<nngvec.at(i).y<<endl;
         cout<<"sumPixel:"<<nngvec.at(i).pixelval<<endl;
         cout<<"numPixel:"<<nngvec.at(i).pixelnum<<endl;
*
* */

/*
if(nngvec.at(i).x==nngvec.at(i+1).y&&nngvec.at(i).y==nngvec.at(i+1).x){
    nngpq.push(nngvec.at(i));
    

   } 




}

*/
cout<<"----------------------------------"<<endl;
//cout<<"vector size:"<<nngvec.size()<<endl;






/*
int ngnum=0;
nngNode *nngp;
nngp=(nngNode*)malloc(sizeof(nngNode)*ragNode);

//for(int p=0;p<ragNode;p++){

//nngNode(int x, int y, int pixelval,int pixelnum,double ndist) : x(x), y(y), pixelval(pixelval),pixelnum(pixelnum),ndist(ndist) {}

for(int i=0;i<id;i++){
     nngNode minimavalue=nngNode(i,0,pixelSum[i],pixelNum[i],neighDist(i,0));
   for(int j=0;j<id;j++){
     if(neighDist(i,j)>0&&i!=j){	   
      if(minimavalue.ndist>neighDist(i,j)){
        // minimavalue.ndist=neighDist(i,j);
        // minimavalue.x=i;
        // minimavalue.y=j;
	// minimavalue.pixelval=pixelSum[i];
	// minimavalue.pixelnum=pixelNum[i];
	 minimavalue=nngNode(i,j,pixelSum[i],pixelNum[i],neighDist(i,j));
         nngp[ngnum]=nngNode(minimavalue.x,minimavalue.y,pixelSum[minimavalue.x],pixelNum[minimavalue.x],neighDist(minimavalue.x,minimavalue.y));
*/	 /*	 
	 nngp[ngnum].ndist=minimavalue.ndist;
         nngp[ngnum].x=minimavalue.x;
         nngp[ngnum].y=minimavalue.y;
         nngp[ngnum].pixelval=minimavalue.pixelval;
         nngp[ngnum].pixelnum=minimavalue.pixelnum; 
	 */
  /*      }
        if(j==(id-1)){
         ngnum++;		
	}	
       
       }
      }
    */  
     /*
        if(j==(id-1)){
         
         nngp[ngnum].ndist=minimavalue.ndist;
         nngp[ngnum].x=minimavalue.x;
         nngp[ngnum].y=minimavalue.y;
         nngp[ngnum].pixelval=minimavalue.pixelval;
         nngp[ngnum].pixelnum=minimavalue.pixelnum; 
         ngnum++;
	}
      }
      */
     /*
      //if(minimavalue.ndist>0){
         nngp[ngnum].ndist=minimavalue.ndist;
         nngp[ngnum].x=minimavalue.x;
         nngp[ngnum].y=minimavalue.y;
         nngp[ngnum].pixelval=minimavalue.pixelval;
         nngp[ngnum].pixelnum=minimavalue.pixelnum;
	 cout<<"-----------nngp----------------"<<endl;
	 cout<<"distanceValue:"<<nngp[ngnum].ndist<<endl;
	 cout<<"x:"<<nngp[ngnum].x<<endl;
	 cout<<"y:"<<nngp[ngnum].y<<endl;
	 cout<<"sumPixel:"<<nngp[ngnum].pixelval<<endl;
	 cout<<"numPixel:"<<nngp[ngnum].pixelnum<<endl;


	 cout<<"###reverse distanceValue:###"<<neighDist(nngp[ngnum].y,nngp[ngnum].x)<<endl;
         cout<<"x:"<<nngp[ngnum].y<<endl;
         cout<<"y:"<<nngp[ngnum].x<<endl;
         cout<<"sumPixel:"<<pixelSum[nngp[ngnum].y]<<endl;
         cout<<"numPixel:"<<pixelNum[nngp[ngnum].y]<<endl;
         ngnum++;
     // }
*/
   //free(minimavalue);
//}
//cout<<"----------------------"<<endl;
//cout<<"ngnum"<<ngnum<<endl;
/*

for(int i=0;i<ngnum;i++){
      cout<<"-----------nngp----------------"<<endl;
      cout<<"distanceValue:"<<nngp[i].ndist<<endl;
      cout<<"x:"<<nngp[i].x<<endl;
      cout<<"y:"<<nngp[i].y<<endl;
      cout<<"sumPixel:"<<nngp[i].pixelval<<endl;
      cout<<"numPixel:"<<nngp[i].pixelnum<<endl;
}


cout<<"----------------------"<<endl;
//cout<<"ngnum"<<ngnum<<endl;
//}

*/

/*#pragma omp parallel for 
for(int i=0;i<id;i++){
  for(int j=0;j<id;j++){
    
   cout<<neighDist[i][j]<<endl;
    
   }
}
*/
/*
string filename32="./test32.txt";
ofstream fout32(filename32);
for(int i = 0; i < id; i++) {
    for(int j = 0; j < id; j++) {
       fout32<<neighDist(i,j)<<",";
       //if(neighDis(i,j)<0){
       //cout<<"negtive"<<endl;
      }
       fout32<<endl;
  }
   
fout32.close();
*/
 // nngNode nngp[ragNode];
/*
double __restrict ***nngGraph=new double**[ragNode];
for(int i=0;i<ragNode;i++){
   nngGraph[i]=new double*[1];
   for(int j=0;j<1;j++){
    nngGraph[i][j]=new double[5];
     nngGraph[i][j][0]=0;//cordx
     nngGraph[i][j][1]=0;//cordy
     nngGraph[i][j][2]=0;//pixelNum
     nngGraph[i][j][3]=0;//pixelSum
     nngGraph[i][j][4]=0;//Dist
   
   }

}


*/
/*
  int cmount=0;
  nngNode tempnode[id];
  for(int i=0;i<id;i++){
   nngNode tempnode;
   tempnode[i].cordx=0;
   tempnode[i].cordy=0;
   tempnode[i].pixelnngVal=0;
   tempnode[i].pixelnngNum=0;
   tempnode[i].nngDist=0.0;
   for(int j=0;j<id;j++){
    if(neighDis(i,j)>0){
       if(tempnode[i].nngDist==0){
         tempnode[i].nngDist=neighDis(i,j);

	 }

        if(tempnode[i].nngDist!=0){
		if(tempnode[i].nngDist>neighDis(i,j)){
		   tempnode[i].nngDist=neighDis(i,j);
		   tempnode[i].cordx=i;
                   tempnode[i].cordy=j;
                   tempnode[i].pixelnngVal=pixelSum[i];
                   tempnode[i].pixelnngNum=pixelNum[i];
                    
                  //nngGraph[i][j][0]=i;
                  //nngGraph[i][j][1]=j;
                  //nngGraph[i][j][2]=pixelNum[i];
                  //nngGraph[i][j][3]=pixelSum[i];
                  //nngGraph[i][j][4]=neighDis(i,j);
         }
       }
    }
   }
   if(tempnode[i].nngDist!=0){
      nngp[cmount].nngDist = tempnode[i].nngDist;
      nngp[cmount].cordx = tempnode[i].cordx;
      nngp[cmount].cordy = tempnode[i].cordy;
      nngp[cmount].pixelnngVal = tempnode[i].pixelnngVal;
      nngp[cmount].pixelnngNum = tempnode[i].pixelnngNum;
      cmount++; 
      cout<<"Distance:"<<nngp[cmount].cordx<<":"<<nngp[cmount].nngDist<<endl;      
   }
}
*/
/*
string filename33="./test33.txt";
ofstream fout33(filename33);
for(int i = 0; i < ragNode; i++) {
 //   for(int j = 0; j < id; j++) {
    // fout32<<neighDis(i,j)<<",";
    // if(neighDis(i,j)<0){
     //cout<<"negtive"<<endl;
 
   //   }
    // }
    //
         fout33<<"nngp:"<<i<<": ("<<nngp[i].cordx<<","<<nngp[i].cordy<<"):"<<"distance"<<nngp[i].nngDist<<endl;
         

    }
 fout33.close();
*/







//int minVal=0;
//
//
//
//
//
/*
for(int i = 0; i < id; i++) {
    for(int j = 0; j < id; j++) {
     //fout32<<neighDis(i,j)<<",";
     if(neighDis(i,j)<0){
     cout<<"negtive"<<endl;

      }
     }
}
*/
 /*
struct nngNode nngGraph[ragNode];
int index=0;


#pragma omp parallel for
for(int i=0;i<id;i++){
 nngNode minVal=nngNode();
 for(int j=0;j<id;j++){
  //  int minVal=0;
  if(neighDis(i,j)>=0){
    if(minVal.nngDist==0){
       minVal.nngDist=neighDis(i,j);
       minVal.cordx=i;
       minVal.cordy=j;
       minVal.pixelnngVal=pixelSum[i];
       minVal.pixelnngNum=pixelNum[i];
      }
      else {
         if(minVal.nngDist>=neighDis(i,j)){
            minVal.nngDist=neighDis(i,j);
            minVal.cordx=i;
            minVal.cordy=j;
            minVal.pixelnngVal=pixelSum[i];
            minVal.pixelnngNum=pixelNum[i];
          }  

      }
    }
  }      
       if(minVal.nngDist>0){
          nngGraph[index].nngDist = minVal.nngDist;
          nngGraph[index].cordx = minVal.cordx;
          nngGraph[index].cordy = minVal.cordy;
          nngGraph[index].pixelnngVal = minVal.pixelnngVal;
          nngGraph[index].pixelnngNum = minVal.pixelnngNum;
          index++;
	  cout<<"over 0"<<endl;

          }
         if(minVal.nngDist==0){
           nngGraph[index].nngDist = minVal.nngDist;
           nngGraph[index].cordx = minVal.cordx;
           nngGraph[index].cordy = minVal.cordy;
           nngGraph[index].pixelnngVal = minVal.pixelnngVal;
           nngGraph[index].pixelnngNum = minVal.pixelnngNum;
	   index++;
	 }
}

//struct nngNode nngGraph[rage]

cout<<nngGraph[3].nngDist<<endl;
cout<<nngGraph[233].nngDist<<endl;
cout<<nngGraph[500].nngDist<<endl;
cout<<nngGraph[335].nngDist<<endl;
cout<<nngGraph[78].nngDist<<endl;
*/
//int nngrows=ragNode/2;
//cout<<"RAG nodes:"<<ragNode/2<<endl;


//Array2D<double>nngGraph(nngrows, nngrows,0.0);
       cv::addWeighted(duplImage,0.1,testDuplicate,0.9,0,dstImage);

    
        duplImage.release();
        testDuplicate.release();
        return dstImage;
    }
