#include<iostream>
#include<cstring>
using namespace std;

int main(){

int a[10]={0};
cout<<sizeof(a)/sizeof(int)<<endl;
memset(a,-1,sizeof(a));

for(int i=0;i<10;i++){

cout<<i<<":"<<a[i]<<endl;
}


}
