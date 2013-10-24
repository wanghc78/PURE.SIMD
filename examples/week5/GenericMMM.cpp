#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <timing.h>
#include <gsimd.h>
#include <stdio.h>
#include <sys/time.h>
//#include <nmmintrin.h>

#define M_SIZE 256
#define ITERATIONS 100

float A[M_SIZE*M_SIZE];
float B[M_SIZE*M_SIZE];
float C_sol[M_SIZE*M_SIZE];
float C[M_SIZE*M_SIZE];

static inline double get_sec()
{
  struct timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

/*
 *
 * Transposes A in place.
 *
 */
void matrixTranspose(float A[M_SIZE*M_SIZE]) {

	int i,j;
	float t;

	for (i=0; i<M_SIZE; i++) {
		for (j=i; j<M_SIZE; j++) {
			t = A[i*M_SIZE+j];
			A[i*M_SIZE+j] = A[j*M_SIZE+i];
			A[j*M_SIZE+i] = t;
		}
	}
}

/* Performs C = A(B^T) as a 2D matrix multiply
 * This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */
void matrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float sum;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j++) {
			sum = 0.0;
			for (k=0; k<M_SIZE; k++) {
				sum += A[i*M_SIZE+k] * B[j*M_SIZE+k];
			}
			C[i*M_SIZE+j] = sum;
		}
	}
}


typedef svec<4,float> vfloat;

void svec4_matrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	vfloat tempI, tempB, sum1, sum2, sum3, sum4;

	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j += 4) {
			vfloat sum1(0.0f);
			vfloat sum2(0.0f);
			vfloat sum3(0.0f);
			vfloat sum4(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempI = vfloat::load((vfloat *)(A + i*M_SIZE + k));
				tempB = vfloat::load((vfloat *)(B + j*M_SIZE + k));
				sum1 = sum1 + tempI*tempB;
				tempB = vfloat::load((vfloat *)(B + (j + 1)*M_SIZE + k));
				sum2 = sum2 + tempI*tempB;
				tempB = vfloat::load((vfloat *)(B + (j + 2)*M_SIZE + k));
				sum3 = sum3 + tempI*tempB;
				tempB = vfloat::load((vfloat *)(B + (j + 3)*M_SIZE + k));
				sum4 = sum4 + tempI*tempB;
			}
			/* I don't know the equivalent code for this in generic
			
			   sum1 = _mm_hadd_ps(sum1,sum2);
			   sum2 = _mm_hadd_ps(sum3,sum4);
			   sum1 = _mm_hadd_ps(sum1,sum2);
			*/
			float working_sums[4];
			float sums[4];

			sum1.store((vfloat *)(working_sums));
			sums[0] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			sum2.store((vfloat *)(working_sums));
			sums[1] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			sum3.store((vfloat *)(working_sums));
			sums[2] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			sum4.store((vfloat *)(working_sums));
			sums[3] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			sum1 = vfloat::load((vfloat *)(sums));
			sum1.store((vfloat *)(C + i*M_SIZE + j));
		}
	}
}

void svec4_matrixMultiply_ptr(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	vfloat tempI, tempB, sum1, sum2, sum3, sum4;

	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j += 4) {
			vfloat sum1(0.0f);
			vfloat sum2(0.0f);
			vfloat sum3(0.0f);
			vfloat sum4(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempI = *(vfloat *)(A + i*M_SIZE + k);
				tempB = *(vfloat *)(B + j*M_SIZE + k);
				sum1 = sum1 + tempI*tempB;
				tempB = *(vfloat *)(B + (j + 1)*M_SIZE + k);
				sum2 = sum2 + tempI*tempB;
				tempB = *(vfloat *)(B + (j + 2)*M_SIZE + k);
				sum3 = sum3 + tempI*tempB;
				tempB = *(vfloat *)(B + (j + 3)*M_SIZE + k);
				sum4 = sum4 + tempI*tempB;
			}
			/* I don't know the equivalent code for this in generic
			
			   sum1 = _mm_hadd_ps(sum1,sum2);
			   sum2 = _mm_hadd_ps(sum3,sum4);
			   sum1 = _mm_hadd_ps(sum1,sum2);
			*/
			float working_sums[4];
			float sums[4];

			*(vfloat *)(working_sums) = sum1;
			sums[0] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			*(vfloat *)(working_sums) = sum2;
			sums[1] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			*(vfloat *)(working_sums) = sum3;
			sums[2] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			*(vfloat *)(working_sums) = sum4;
			sums[3] = working_sums[0] + working_sums[1] + working_sums[2] + working_sums[3];

			sum1 = *(vfloat *)(sums);
			*(vfloat *)(C + i*M_SIZE + j) = sum1;
		}
	}
}

#ifdef __SSE4_2__
/* 
 * Performs C = A(B^T) as a 2D matrix multiply 
 * This is done using SIMD instructions
*/
void SIMD_matrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	__m128 tempI, tempB, sum1, sum2, sum3, sum4;

	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j += 4) {
			sum1 = _mm_set1_ps(0.0f);
			sum2 = _mm_set1_ps(0.0f);
			sum3 = _mm_set1_ps(0.0f);
			sum4 = _mm_set1_ps(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempI = _mm_loadu_ps(A + i*M_SIZE + k);
				tempB = _mm_loadu_ps(B + j*M_SIZE + k);
				sum1 = _mm_add_ps(sum1,_mm_mul_ps(tempI,tempB));
				tempB = _mm_loadu_ps(B + (j + 1)*M_SIZE + k);
				sum2 = _mm_add_ps(sum2,_mm_mul_ps(tempI,tempB));
				tempB = _mm_loadu_ps(B + (j + 2)*M_SIZE + k);
				sum3 = _mm_add_ps(sum3,_mm_mul_ps(tempI,tempB));
				tempB = _mm_loadu_ps(B + (j + 3)*M_SIZE + k);
				sum4 = _mm_add_ps(sum4,_mm_mul_ps(tempI,tempB));
			}
			sum1 = _mm_hadd_ps(sum1,sum2);
			sum2 = _mm_hadd_ps(sum3,sum4);
			sum1 = _mm_hadd_ps(sum1,sum2);
			_mm_storeu_ps(C + i*M_SIZE + j,sum1);
		}
	}
}
#endif

int check_mat(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE]) {
	int i,j;
	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			if (abs(A[i*M_SIZE + j] - B[i*M_SIZE + j]) > 0.001) {
				return 0;
			}
		}
	}
	return 1;
}


int main() {
	double start_time, end_time;
	double seq_time;
	int i,j,k;
	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			A[i*M_SIZE + j] = (rand() % 256) - 128;
			B[i*M_SIZE + j] = (rand() % 256) - 128;
		}
	}
	
	// First, transpose B
	matrixTranspose(B);

	
	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		matrixMultiply(A,B,C_sol);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	printf("Sequential version finished,\ttime %f\n", seq_time);


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		svec4_matrixMultiply(A,B,C);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		printf("svec4 version finished,\ttime %f\n",seq_time);
	}
	else {
		printf("svec4 version failed\n");
	}

	
	// This one isn't working
	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		svec4_matrixMultiply_ptr(A,B,C);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		printf("svec4 ptr version finished,\ttime %f\n",seq_time);
	}
	else {
		printf("svec4 ptr version failed\n");
	}


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMD_matrixMultiply(A,B,C);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		printf("SIMD version finished,\ttime %f\n",seq_time);
	}
	else {
		printf("SIMD version failed\n");
	}

	return 0;
}
