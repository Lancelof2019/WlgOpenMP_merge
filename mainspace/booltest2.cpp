#include<iostream>
using namespace std;
int main()
{
  /*  int arr1[5][5] = { { 0, 1, 2, 3, 4 },
                       { 2, 3, 4, 5, 6 },
                       { 4, 5, 6, 7, 8 },
                       { 5, 4, 3, 2, 6 },
                       { 2, 5, 4, 3, 1 } };
 */
		       int* arr2[5][5];

    // Initialising each element of the
    // pointer array with the address of
  /*  // element present in the other array
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            arr2[i][j] = &arr1[i][j];
        }
    }
*/


//int *p=&arr1[0][2];
int m=3;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            arr2[i][j] =&m ;
        }
    }


    // Printing the array using
    // the array of pointers
    cout<<"The values are"<<endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout<< *arr2[i][j];
        }
        cout<<endl;
    }

    return 0;
}
