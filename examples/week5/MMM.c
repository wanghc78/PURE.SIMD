#include <stdio.h>
#include <sys/time.h>
#include <nmmintrin.h>

#define M_SIZE 256 //256

float A[M_SIZE*M_SIZE];
float B[M_SIZE*M_SIZE];
float C[M_SIZE*M_SIZE];

static inline double get_sec()
{
  struct timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

/* Performs C = A(B^T) as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */

void matrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float prod[M_SIZE];
	float sum;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j++) {
			sum = 0.0;
			for (k=0; k<M_SIZE; k++) {
				prod[k] = A[i*M_SIZE+k] * B[j*M_SIZE+k];
			}
			for (k=0; k<M_SIZE; k++) {
				sum = sum + prod[k];
			}
			C[i*M_SIZE+j] = sum;
		}
	}
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

/* 
 * Performs C = A(B^T) as a 2D matrix multiply 
 * This is done using SIMD instructions
*/
void SIMDmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	__m128 tempA, tempB, sum;

	float total_sum[4];
	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			sum = _mm_set1_ps(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempA = _mm_loadu_ps(A + i*M_SIZE + k);
				tempB = _mm_loadu_ps(B + j*M_SIZE + k);
				sum = _mm_add_ps(sum,_mm_mul_ps(tempA,tempB));
			}
			_mm_storeu_ps(total_sum,sum);
			C[i*M_SIZE+j] = total_sum[0] + total_sum[1] + total_sum[2] + total_sum[3];
		}
	}
}

/*
 * Transposes A in place using SIMD instructions.
 */
void SIMDmatrixTranspose(float A[M_SIZE*M_SIZE]) {

	void swap(int a, int b, int size) {
		__m128 temp1, temp2;
		int i,j;
		for (i = 0; i < size; i ++) {
			for (j = 0; j < size/4; j ++) {
				float *swap1 = A + (a + i)*M_SIZE + b + size + j*4;
				float *swap2 = A + (a + i + size)*M_SIZE + b + j*4;
				temp1 = _mm_loadu_ps(swap1);
				temp2 = _mm_loadu_ps(swap2);
				_mm_storeu_ps(swap1,temp2);
				_mm_storeu_ps(swap2,temp1);
			}
		}
	}

	void recursiveTranspose(int a, int b, int size) {
		if (size > 4) {
			int new_size = size/2;
			swap(a,b,new_size);
			recursiveTranspose(a,b,new_size);
			recursiveTranspose(a + new_size,b,new_size);
			recursiveTranspose(a,b + new_size,new_size);
			recursiveTranspose(a + new_size,b + new_size,new_size);
		}
		else {
			__m128 r1,r2,r3,r4,s1,s2,s3,s4;
			float *swap1 = A + a*M_SIZE + b;
			float *swap2 = A + (a + 1)*M_SIZE + b;
			float *swap3 = A + (a + 2)*M_SIZE + b;
			float *swap4 = A + (a + 3)*M_SIZE + b;
			
			r1 = _mm_loadu_ps(swap1);
			r2 = _mm_loadu_ps(swap2);
			r3 = _mm_loadu_ps(swap3);
			r4 = _mm_loadu_ps(swap4);
			
			s1 = _mm_unpacklo_ps(r1,r3);
			s2 = _mm_unpacklo_ps(r2,r4);
			s3 = _mm_unpackhi_ps(r1,r3);
			s4 = _mm_unpackhi_ps(r2,r4);
			
			r1 = _mm_unpacklo_ps(s1,s2);
			r2 = _mm_unpackhi_ps(s1,s2);
			r3 = _mm_unpacklo_ps(s3,s4);
			r4 = _mm_unpackhi_ps(s3,s4);
			
			_mm_storeu_ps(swap1,r1);
			_mm_storeu_ps(swap2,r2);
			_mm_storeu_ps(swap3,r3);
			_mm_storeu_ps(swap4,r4);
		}
	}

	recursiveTranspose(0,0,M_SIZE);
}

void print_mat(float A[M_SIZE*M_SIZE]) {
	int i,j;
	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			printf("%d\t",(int)A[i*M_SIZE + j]);
		}
		printf("\n");
	}
}

#define ITERATIONS 10

int main() {
	double start_time, end_time;
	int i,j,k;
	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			A[i*M_SIZE + j] = (rand() % 256) - 128;
			B[i*M_SIZE + j] = (rand() % 256) - 128;
		}
	}

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		matrixTranspose(B);
		matrixMultiply(A,B,C);
	}
	end_time = get_sec();
	printf("Sequential version finished, time %f\n", (end_time-start_time));

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		matrixTranspose(B);
		SIMDmatrixMultiply(A,B,C);
	}
	end_time = get_sec();
	printf("SIMD version finished, time %f\n", (end_time-start_time));

	return 0;
}
