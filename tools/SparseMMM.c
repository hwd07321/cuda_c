#include<stdio.h>
#include<stdlib.h>
// #include<iostream>
#include<time.h>
// #include<immintrin.h>

// COO matrix type
typedef struct COOTriplet {
    int *values;
    int *row_indices;
    int *col_indices;
    int num_nonzeros;
} COOMatrix;

typedef struct {
    int *values;
    int *col_indices;
    int *row_ptr;
    int num_rows;
    int num_cols;
    int num_nonzeros;
} CSRMatrix;

typedef struct {
    int *values;
    int *row_indices;
    int *col_ptr;
    int num_rows;
    int num_cols;
    int num_nonzeros;
}CSCMatrix;

COOMatrix ConvertCOO(int **matrix, int row, int col){
    COOMatrix coo;
    coo.num_nonzeros = 0;
    for(int i = 0 ; i < row ; i++){
        for(int j = 0 ; j < col ; j++){
            if(matrix[i][j] != 0){
                coo.num_nonzeros++;
            }
        }
    }
    coo.values = (int *)malloc(coo.num_nonzeros * sizeof(int));
    coo.row_indices = (int *)malloc(coo.num_nonzeros * sizeof(int));
    coo.col_indices = (int *)malloc(coo.num_nonzeros * sizeof(int));

    int k = 0;
    for(int i = 0 ; i < row ; i++){
        for(int j = 0 ; j < col ; j++){
            if(matrix[i][j] != 0){
                coo.values[k] = matrix[i][j];
                coo.row_indices[k] = i;
                coo.col_indices[k] = j;
                k++;
            }
        }
    }
    return coo;
}

CSRMatrix ConvertCSR(int **matrix, int row, int col){
    CSRMatrix csr;
    csr.num_nonzeros = 0;
    csr.num_cols = col;
    csr.num_rows = row;
    for(int i = 0 ; i < row ; i++){
        for(int j = 0 ; j < col ; j++){
            if(matrix[i][j] != 0){
                csr.num_nonzeros++;
            }
        }
    }

    csr.values = (int *)malloc(csr.num_nonzeros * sizeof(int));
    csr.col_indices = (int *)malloc(csr.num_nonzeros * sizeof(int));
    csr.row_ptr = (int *)malloc((csr.num_rows + 1) * sizeof(int));
    int k = 0;
    for(int i = 0 ; i < row ; i++){
        csr.row_ptr[i] = k;
        printf("k is %d\n", k);
        for(int j = 0 ; j < col ; j++){
            if(matrix[i][j] != 0){
                csr.values[k] = matrix[i][j];
                csr.col_indices[k] = j;
                k++;
            }
        }
    }
    csr.row_ptr[row] = csr.num_nonzeros;
    return csr;
}

CSCMatrix ConvertCSC(int **matrix, int row, int col){
    CSCMatrix csc;
    csc.num_nonzeros = 0;
    csc.num_rows = row;
    csc.num_cols = col;
    for(int i = 0 ; i < row ; i++){
        for(int j = 0 ; j < col ; j++){
            if(matrix[i][j] != 0){
                csc.num_nonzeros++;
            }
        }
    }

    csc.values = (int *)malloc(csc.num_nonzeros * sizeof(int));
    csc.row_indices = (int *)malloc(csc.num_nonzeros * sizeof(int));
    csc.col_ptr = (int *)malloc((csc.num_cols+1) * sizeof(int));

    int k;

    for(int j = 0 ; j < col ; j++){
        csc.col_ptr[j] = k;
        for(int i = 0 ; i < row ; i++){
            if(matrix[i][j] != 0){
                csc.values[k] = matrix[i][j];
                csc.row_indices[k] = i;
                k++;
            }
        }
    }
    csc.col_ptr[col] = csc.num_nonzeros;
    return csc;
}




int main(){
    const int rows = 4, cols = 5;
    int matrix[rows][cols] = {
        {1, 0, 3, 0, 0},
        {0, 4, 0, 0, 6},
        {0, 0, 0, 0, 0},
        {7, 0, 0, 8, 9}
    };
    int nnz; // 非零元素数量

    int *matrix_ptr[rows];
    for(int i = 0 ; i < rows ; i++){
        matrix_ptr[i] = matrix[i];
    }

    // COO
    COOMatrix coo = ConvertCOO(matrix_ptr, rows, cols);
    
    printf("Values: ");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        printf("%d ", coo.values[i]);
    }
    printf("\n");

    printf("Row Indices: ");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        printf("%d ", coo.row_indices[i]);
    }
    printf("\n");

    printf("Column Indices: ");
    for (int i = 0; i < coo.num_nonzeros; i++) {
        printf("%d ", coo.col_indices[i]);
    }
    printf("\n");
    
    // CSR
    CSRMatrix csr = ConvertCSR(matrix_ptr, rows, cols);

    printf("Values: ");
    for (int i = 0; i < csr.num_nonzeros; i++) {
        printf("%d ", csr.values[i]);
    }
    printf("\n");

    printf("Column Indices: ");
    for (int i = 0; i < csr.num_nonzeros; i++) {
        printf("%d ", csr.col_indices[i]);
    }
    printf("\n");

    printf("Row Pointer: ");
    for (int i = 0; i < csr.num_rows + 1; i++) {
        printf("%d ", csr.row_ptr[i]);
    }
    printf("\n");

    return 0;
}


