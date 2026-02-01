
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"

float randn() {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}


int CreateArmCublas(cublasHandle_t handle, int d, float lambda, float **d_A, float **d_B) {
    float *h_A = (float*)malloc(d * d * sizeof(float));
    float *h_B = (float*)malloc(d * sizeof(float));

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            h_A[j * d + i] = (i == j) ? lambda : 0.0f;
        }
        h_B[i] = 0.0f;
    }

    cudaMalloc((void**)d_A, d * d * sizeof(float));
    cudaMalloc((void**)d_B, d * sizeof(float));

    cudaMemcpy(*d_A, h_A, d * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, h_B, d * sizeof(float), cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    return 0;
}

int makeArmsCublas(cublasHandle_t handle, int n_arms, int d, float lambda, float ***d_As, float ***d_Bs) {
    *d_As = (float**)malloc(n_arms * sizeof(float*));
    *d_Bs = (float**)malloc(n_arms * sizeof(float*));

    for (int arm = 0; arm < n_arms; arm++) {
        CreateArmCublas(handle, d, lambda, &((*d_As)[arm]), &((*d_Bs)[arm]));
    }

    return 0;
}


int makeSyntheticContextVectorCublas(cublasHandle_t handle, int dims, float **d_x) {
    float *h_x = (float*)malloc(dims * sizeof(float));
    for (int i = 0; i < dims; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc((void**)d_x, dims * sizeof(float));
    cudaMemcpy(*d_x, h_x, dims * sizeof(float), cudaMemcpyHostToDevice);

    free(h_x);
    return 0;
}


int outerProductCublas(cublasHandle_t handle, float *d_a, float *d_b, float *d_result, int len_a, int len_b) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // d_result = alpha * d_a * d_b^T + beta * d_result
    cublasSger(handle, len_a, len_b, &alpha, d_a, 1, d_b, 1, d_result, len_a);

    return 0;
}


int makeTrueThetaCublas(float **trueTheta, int n_arms, int n_dimensions) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int k = 0; k < n_arms; k++) {
        float *d_v;
        cudaMalloc((void**)&d_v, n_dimensions * sizeof(float));

        float *h_v = (float*)malloc(n_dimensions * sizeof(float));
        for (int i = 0; i < n_dimensions; i++) {
            h_v[i] =randn(); 
        }

        cudaMemcpy(d_v, h_v, n_dimensions * sizeof(float), cudaMemcpyHostToDevice);
        trueTheta[k] = d_v;

        free(h_v);
    }

    cublasDestroy(handle);
    return 0;
}

int invertMatrixCuSolver(cusolverDnHandle_t solver, int d,
            float *d_A,      
            float *d_A_inv,   
            float *d_work,    
            int work_size,   
            int *d_info)
{
    int *d_ipiv;
    cudaMalloc((void**)&d_ipiv, d * sizeof(int));

    cusolverDnSgetrf(solver, d, d, d_A, d, d_work, d_ipiv, d_info);

    // signature: solver, trans, n, nrhs, A, lda, ipiv, B, ldb, info
    cusolverDnSgetrs(solver, CUBLAS_OP_N, d, d, d_A, d, d_ipiv, d_A_inv, d, d_info);

    cudaFree(d_ipiv);
    return 0;
}

float dotProductCublas(cublasHandle_t handle, float *d_a, float *d_b, int d) {
    float res = 0.0f;
    cublasSdot(handle, d, d_a, 1, d_b, 1, &res);
    return res;
}


void matVecCublas(cublasHandle_t handle, float *d_M, float *d_v, float *d_out, int d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // d_out = alpha * d_M * d_v + beta * d_out
    cublasSgemv(handle, CUBLAS_OP_T, d, d, &alpha, d_M, d, d_v, 1, &beta, d_out, 1);
}




int main (void){
    // 
    srand(42);

    const int n_trials = 1000;
    const int n_arms = 50;
    const int n_dimensions = 300;
    const float lambda = 0.1f;
    const float alpha = 1.0f;

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    cublasCreate(&cublas_handle);
    cusolverDnCreate(&solver_handle);

    float **trueTheta = (float**)malloc(n_arms * sizeof(float*));
    makeTrueThetaCublas(trueTheta, n_arms, n_dimensions);

    float **d_As, **d_Bs;
    makeArmsCublas(cublas_handle, n_arms, n_dimensions, lambda, &d_As, &d_Bs); 

    float *d_X =NULL;
    float *d_A_copy = NULL;
    float *d_A_inv = NULL;
    float *d_theta = NULL;
    float *d_temp = NULL;
    float *d_outer = NULL;
    float *d_bupdate = NULL;
    int *d_info = NULL;

    cudaMalloc((void**)&d_X, n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_A_copy, n_dimensions * n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_A_inv, n_dimensions * n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_theta, n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_temp, n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_outer, n_dimensions * n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_bupdate, n_dimensions * sizeof(float));
    cudaMalloc((void**)&d_info, sizeof(int));

    int work_size = 0;
    cusolverDnSgetrf_bufferSize(solver_handle, n_dimensions, n_dimensions, (float*)NULL, n_dimensions, &work_size);

    int work_size2 = 0;
    cusolverDnSgetrf_bufferSize(solver_handle, n_dimensions, n_dimensions, (float*)NULL, n_dimensions, &work_size2);

    if (work_size2 > work_size) work_size = work_size2;

    float *d_work = NULL;
    cudaMalloc((void**)&d_work, work_size * sizeof(float));

    float *h_pk = (float*)malloc(n_arms * sizeof(float));
    float *h_trueRewards = (float*)malloc(n_arms * sizeof(float));

    double cumulative_regret = 0.0;

    float *h_I = (float*)calloc(n_dimensions * n_dimensions, sizeof(float));
    for (int i = 0; i < n_dimensions; i++)
        h_I[i * n_dimensions + i] = 1.0f;


    for (int t = 0; t < n_trials; t++) {

        makeSyntheticContextVectorCublas(cublas_handle, n_dimensions, &d_X);

        // compute p_k for each arm
        for (int k = 0; k < n_arms; k++) {
            // copy A to A_copy
            cudaMemcpy(d_A_copy, d_As[k], n_dimensions * n_dimensions * sizeof(float), cudaMemcpyDeviceToDevice);

            // {
            //     float *h_I = (float*)calloc(n_dimensions * n_dimensions, sizeof(float));
            //     for (int i = 0; i < n_dimensions; i++)
            //         h_I[i * n_dimensions + i] = 1.0f;
            //     cudaMemcpy(d_A_inv, h_I, n_dimensions * n_dimensions * sizeof(float), cudaMemcpyHostToDevice);
            //     free(h_I);
            // }
            cudaMemcpy(d_A_inv, h_I, n_dimensions * n_dimensions * sizeof(float), cudaMemcpyHostToDevice);


            // invert A_copy to A_inv
            invertMatrixCuSolver(solver_handle, n_dimensions, d_A_copy, d_A_inv, d_work, work_size, d_info);

            // compute theta = A_inv * B
            matVecCublas(cublas_handle, d_A_inv, d_Bs[k], d_theta, n_dimensions);

            float mean = dotProductCublas(cublas_handle, d_theta, d_X, n_dimensions);

            // compute p_k = x^T * theta + alpha * sqrt(x^T * A_inv * x)
            matVecCublas(cublas_handle, d_A_inv, d_X, d_temp, n_dimensions);

            float xAx = dotProductCublas(cublas_handle, d_X, d_temp, n_dimensions);
            float uncertainty = sqrtf(fmaxf(xAx, 0.0f));

            h_pk[k] = mean + alpha * uncertainty;

        }

        int selectedArm = 0;
        for (int k = 1; k < n_arms; k++) {
            if (h_pk[k] > h_pk[selectedArm]) {
                selectedArm = k;
            }
        }


        float trueReward = dotProductCublas(cublas_handle, trueTheta[selectedArm], d_X, n_dimensions);
        float noise = randn() * 0.1f;
        float reward = trueReward + noise;

        float bestReward = -INFINITY;
        for (int k = 0; k < n_arms; k++) {
            h_trueRewards[k] = dotProductCublas(cublas_handle, trueTheta[k], d_X, n_dimensions);
            if (h_trueRewards[k] > bestReward) {
                bestReward = h_trueRewards[k];
            }
        }

        float regret = bestReward - trueReward;
        cumulative_regret += regret;

        printf("Trial %d: Selected Arm %d, Reward: %.4f, Cumulative Regret: %.4f, This regret: %.4f\n", t + 1, selectedArm, reward, cumulative_regret, regret);


        outerProductCublas(cublas_handle, d_X, d_X, d_outer, n_dimensions, n_dimensions);
        {
            const float alpha = 1.0f;
            cublasSaxpy(cublas_handle, n_dimensions * n_dimensions, &alpha, d_outer, 1, d_As[selectedArm], 1);
        }

        {
            float scalar = reward;
            cublasSaxpy(cublas_handle, n_dimensions, &scalar, d_X, 1, d_Bs[selectedArm], 1);
        }

    }

    cudaFree(d_X);
    cudaFree(d_A_copy);
    cudaFree(d_A_inv);
    cudaFree(d_theta);
    cudaFree(d_temp);
    cudaFree(d_outer);
    cudaFree(d_bupdate);
    cudaFree(d_info);
    cudaFree(d_work);
    for (int k = 0; k < n_arms; k++) {
        cudaFree(trueTheta[k]);
    }
    free(d_As);
    free(d_Bs);
    free(trueTheta);
    free(h_pk);
    free(h_I);
    free(h_trueRewards);

    cublasDestroy(cublas_handle);
    cusolverDnDestroy(solver_handle);



    return EXIT_SUCCESS;
}