Array2D<int> &lcSwitch(Array<int>&image,Array<int>&lcimage,Array2D<bool>&vistArr){

queue(pixel,vector(pixel)) comprq;

for(int i=0;i<image.rows;i++){
  for(int j=0;j<image.cols;j++){
      for(int h=0;h<8;h++){
	int  nextX=dx[h]+i;
	int  nextY=dy[h]+j;
	  if(image(i,j)>image(nextX,nextY)){
	       Pixel nextpixel=Pixel(-10,i,j);
               comprq.push(nextpixel);
	       lcimage(i,j)=-1;
               break;
	  }

      }
  }
}


int dist=1;


while(!comprq.empty()){

  // Pixel curpixel=comprq.top();
   int curX=comprq.top().x;
   int curY=comprq.top().y;
   if(curpixel!=-2){
     if(!comprq.empty()){
     

      }


   }


  else{
 
      lcimage(curX,curY)=dist;
      int neighnum=0;
      int puzznum=0;
     for(int h=0;h<8;h++){
         int nextX=curX+dx[h];
         int nextY=curY+dy[h];

	 if(nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols) {
           puzznum++;
           continue;
        }
         if((image(curX,curY)==image(nextX,nextY))&&(lc(nextX, nextY)==0)){
         neighnum++;  
         // Pixel nextpixel=Pixel(-10,nextX,nextY);
         // comprq.push(nextPixel);
         // lcimage(nextX,nextY)=-1;
      } 
    }

   if((neighnum+puzznum)==8){
    for(int h=0;h<8;h++){
         int nextX=curX+dx[h];
         int nextY=curY+dy[h];
       if(nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols) {
         continue;
       }
       if((image(curX,curY)==image(nextX,nextY))&&(lc(nextX, nextY)==0)){
         Pixel nextpixel=Pixel(-10,nextX,nextY);
         lcimage(nextX,nextY)=-1;
         comprq.push(nextpixel);
        }
      }
    }
  }
}

for(int i=0;i<image.rows;i++){
  for(int j=0;j<image.cols;j++){
   if(lcimage(i,j)!=0){
   lc(i,j)=dist*image(i,j)+lcimage(i,j)=1;

   }
   else{
    lcimage(i,j)=dist*image(i,j);
   
   }
  }
}

}
