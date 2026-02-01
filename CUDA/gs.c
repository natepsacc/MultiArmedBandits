#include <stdio.h>

#define R1 3
#define C1 3
#define R2 3
#define C2 3


void mulMat(int mat1[][C1], int mat2[][C2]) {
  int res[R1][C2];

  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      res[i][j] = 0;
      for (int k = 0; k < R2; k++) {
        res[i][j] += mat1[i][k] * mat2[k][j];
      }
      printf("%d ", res[i][j]);
    }
    printf("\n");
  }
}


int main() {
  int mat1[R1][C1] = {  {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9} };
  int mat2[C1][C2] = {  {9, 8, 7},
                        {6, 5, 4},
                        {3, 2, 1} };
  mulMat(mat1, mat2);

  return 0;
}