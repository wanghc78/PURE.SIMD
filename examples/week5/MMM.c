#include <stdio.h>
#include <sys/time.h>
#include <nmmintrin.h>

#define M_SIZE 256 //256
#define ITERATIONS 10

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

__m128 gather_float(float *base, unsigned scale, __m128i offsets, __m128 mask) {
	float temp[4];
	__m128 ret;
	__m128 zero = _mm_set1_ps(0.0f);

	ret = _mm_set_ps(*(base + scale*_mm_extract_epi32(offsets,3)),
			 *(base + scale*_mm_extract_epi32(offsets,2)),
			 *(base + scale*_mm_extract_epi32(offsets,1)),
			 *(base + scale*_mm_extract_epi32(offsets,0))
			 );

	return _mm_blendv_ps(ret,zero,mask);
}

void scatter_float(float *base, unsigned scale, __m128i offsets, __m128 v, __m128 mask ) {
	int temp;
	if (_mm_extract_ps(mask,0)) {
		temp = _mm_extract_ps(v,0);
		*(base + scale*_mm_extract_epi32(offsets,0)) = *((float *) &temp);
	}
	if (_mm_extract_ps(mask,1)) {
		temp = _mm_extract_ps(v,1);
		*(base + scale*_mm_extract_epi32(offsets,1)) = *((float *) &temp);
	}
	if (_mm_extract_ps(mask,2)) {
		temp = _mm_extract_ps(v,2);
		*(base + scale*_mm_extract_epi32(offsets,2)) = *((float *) &temp);
	}
	if (_mm_extract_ps(mask,3)) {
		temp = _mm_extract_ps(v,3);
		*(base + scale*_mm_extract_epi32(offsets,3)) = *((float *) &temp);
	}
}

/* Performs C = A(B^T) as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
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

/* Performs C = A(B^T) as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */

void UNROLLmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float sum;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j++) {
			sum = 0.0;
			for (k=0; k<M_SIZE; k += 4) {
				sum += A[i*M_SIZE + k] * B[j*M_SIZE + k];
				sum += A[i*M_SIZE + k + 1] * B[j*M_SIZE + k + 1];
				sum += A[i*M_SIZE + k + 2] * B[j*M_SIZE + k + 2];
				sum += A[i*M_SIZE + k + 3] * B[j*M_SIZE + k + 3];
			}
			C[i*M_SIZE + j] = sum;
		}
	}
}

/* Performs C = A(B^T) as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */

void quad_UNROLLmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float sum1, sum2, sum3, sum4;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j += 4) {
			sum1 = 0.0;
			sum2 = 0.0;
			sum3 = 0.0;
			sum4 = 0.0;
			for (k=0; k<M_SIZE; k += 4) {
				sum1 += A[i*M_SIZE + k] * B[j*M_SIZE + k];
				sum1 += A[i*M_SIZE + k + 1] * B[j*M_SIZE + k + 1];
				sum1 += A[i*M_SIZE + k + 2] * B[j*M_SIZE + k + 2];
				sum1 += A[i*M_SIZE + k + 3] * B[j*M_SIZE + k + 3];
				sum2 += A[i*M_SIZE + k] * B[(j + 1)*M_SIZE + k];
				sum2 += A[i*M_SIZE + k + 1] * B[(j + 1)*M_SIZE + k + 1];
				sum2 += A[i*M_SIZE + k + 2] * B[(j + 1)*M_SIZE + k + 2];
				sum2 += A[i*M_SIZE + k + 3] * B[(j + 1)*M_SIZE + k + 3];
				sum3 += A[i*M_SIZE + k] * B[(j + 2)*M_SIZE + k];
				sum3 += A[i*M_SIZE + k + 1] * B[(j + 2)*M_SIZE + k + 1];
				sum3 += A[i*M_SIZE + k + 2] * B[(j + 2)*M_SIZE + k + 2];
				sum3 += A[i*M_SIZE + k + 3] * B[(j + 2)*M_SIZE + k + 3];
				sum4 += A[i*M_SIZE + k] * B[(j + 3)*M_SIZE + k];
				sum4 += A[i*M_SIZE + k + 1] * B[(j + 3)*M_SIZE + k + 1];
				sum4 += A[i*M_SIZE + k + 2] * B[(j + 3)*M_SIZE + k + 2];
				sum4 += A[i*M_SIZE + k + 3] * B[(j + 3)*M_SIZE + k + 3];
			}
			C[i*M_SIZE + j] = sum1;
			C[i*M_SIZE + j + 1] = sum2;
			C[i*M_SIZE + j + 2] = sum3;
			C[i*M_SIZE + j + 3] = sum4;
		}
	}
}

/* Performs C = A*B as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */

void REGmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float sum;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j++) {
			sum = 0.0;
			for (k=0; k<M_SIZE; k++) {
				sum += A[i*M_SIZE+k] * B[k*M_SIZE+j];
			}
			C[i*M_SIZE+j] = sum;
		}
	}
}

/* Performs C = A*B as a 2D matrix multiply */

/* This is done using the straight forward implementation which takes
 *
 * O(n^3) time. It is not Strausen's alg which runs in O(n^lg7) time.
 */

void unroll_REGmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {

	float* v1;
	float* v2;
	float sum1, sum2, sum3, sum4;
	int i,j,k;

	for (i=0; i<M_SIZE; i++) {
		for (j=0; j<M_SIZE; j += 4) {
			sum1 = 0.0;
			sum2 = 0.0;
			sum3 = 0.0;
			sum4 = 0.0;
			for (k=0; k<M_SIZE; k += 4) {
				sum1 += A[i*M_SIZE + k] * B[k*M_SIZE + j];
				sum1 += A[i*M_SIZE + k + 1] * B[k*M_SIZE + j + 1];
				sum1 += A[i*M_SIZE + k + 2] * B[k*M_SIZE + j + 2];
				sum1 += A[i*M_SIZE + k + 3] * B[k*M_SIZE + j + 3];
				sum2 += A[i*M_SIZE + k] * B[(k + 1)*M_SIZE + j];
				sum2 += A[i*M_SIZE + k + 1] * B[(k + 1)*M_SIZE + j + 1];
				sum2 += A[i*M_SIZE + k + 2] * B[(k + 1)*M_SIZE + j + 2];
				sum2 += A[i*M_SIZE + k + 3] * B[(k + 1)*M_SIZE + j + 3];
				sum3 += A[i*M_SIZE + k] * B[(k + 2)*M_SIZE + j];
				sum3 += A[i*M_SIZE + k + 1] * B[(k + 2)*M_SIZE + j + 1];
				sum3 += A[i*M_SIZE + k + 2] * B[(k + 2)*M_SIZE + j + 2];
				sum3 += A[i*M_SIZE + k + 3] * B[(k + 2)*M_SIZE + j + 3];
				sum4 += A[i*M_SIZE + k] * B[(k + 3)*M_SIZE + j];
				sum4 += A[i*M_SIZE + k + 1] * B[(k + 3)*M_SIZE + j + 1];
				sum4 += A[i*M_SIZE + k + 2] * B[(k + 3)*M_SIZE + j + 2];
				sum4 += A[i*M_SIZE + k + 3] * B[(k + 3)*M_SIZE + j + 3];
			}
			C[i*M_SIZE + j] = sum1;
			C[i*M_SIZE + j + 1] = sum2;
			C[i*M_SIZE + j + 2] = sum3;
			C[i*M_SIZE + j + 3] = sum4;
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
 * Performs C = A*B as a 2D matrix multiply 
 * This is done using SIMD instructions including gather
*/
void SIMDmatrixMultiplyGS(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	__m128 tempI, tempB, sum1, sum2, sum3, sum4;
	__m128i offsets;
	__m128 mask = _mm_set1_ps(1.0f);
	__m128i inc = _mm_set1_epi32(4);

	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j += 4) {
			sum1 = _mm_set1_ps(0.0f);
			sum2 = _mm_set1_ps(0.0f);
			sum3 = _mm_set1_ps(0.0f);
			sum4 = _mm_set1_ps(0.0f);
			offsets = _mm_set_epi32(3,2,1,0);
			for (k = 0; k < M_SIZE; k += 4) {
				tempI = _mm_loadu_ps(A + i*M_SIZE + k);
				tempB = gather_float(B + j,M_SIZE,offsets,mask);
				sum1 = _mm_add_ps(sum1,_mm_mul_ps(tempI,tempB));
				tempB = gather_float(B + j + 1,M_SIZE,offsets,mask);
				sum2 = _mm_add_ps(sum2,_mm_mul_ps(tempI,tempB));
				tempB = gather_float(B + j + 2,M_SIZE,offsets,mask);
				sum3 = _mm_add_ps(sum3,_mm_mul_ps(tempI,tempB));
				tempB = gather_float(B + j + 3,M_SIZE,offsets,mask);
				sum4 = _mm_add_ps(sum4,_mm_mul_ps(tempI,tempB));
				offsets = _mm_add_epi32(offsets,inc);
			}
			sum1 = _mm_hadd_ps(sum1,sum2);
			sum2 = _mm_hadd_ps(sum3,sum4);
			sum1 = _mm_hadd_ps(sum1,sum2);
			_mm_storeu_ps(C + i*M_SIZE + j,sum1);
		}
	}
}

/* 
 * Performs C = A(B^T) as a 2D matrix multiply 
 * This is done using SIMD instructions
*/
void SIMDmatrixMultiplyQuad(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
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

/* 
 * Performs C = A(B^T) as a 2D matrix multiply 
 * This is done using SIMD instructions
*/
void SIMDmatrixMultiplyNew(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	__m128 tempA, tempB, sum;

	int i,j,k,temp;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			sum = _mm_set1_ps(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempA = _mm_loadu_ps(A + i*M_SIZE + k);
				tempB = _mm_loadu_ps(B + j*M_SIZE + k);
				sum = _mm_add_ps(sum,_mm_mul_ps(tempA,tempB));
			}
			sum = _mm_hadd_ps(sum,sum);
			sum = _mm_hadd_ps(sum,sum);
			temp = _mm_extract_ps(sum,0);
			C[i*M_SIZE+j] = *((float *) &temp);
		}
	}
}

void SIMDmatrixMultiply(float A[M_SIZE*M_SIZE], float B[M_SIZE*M_SIZE], float C[M_SIZE*M_SIZE]) {
	__m128 tempA, tempB, sum;

	float total[4];
	int i,j,k;

	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			sum = _mm_set1_ps(0.0f);
			for (k = 0; k < M_SIZE; k += 4) {
				tempA = _mm_loadu_ps(A + i*M_SIZE + k);
				tempB = _mm_loadu_ps(B + j*M_SIZE + k);
				sum = _mm_add_ps(sum,_mm_mul_ps(tempA,tempB));
			}
			_mm_storeu_ps(total,sum);
			C[i*M_SIZE+j] = total[0] + total[1] + total[2] + total[3];
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


// if theres a second argument, it writes a file instead of printing to shell
int main(int argc, char **argv) {
	FILE *f;
	if (argc > 1) {
		char name[8];
		sprintf(name,"%d.txt",M_SIZE);
		f = fopen(name,"w");
	}
	else {
		f = stdout;
	}
	double start_time, end_time;
	double seq_time;
	double new_time;
	int i,j,k;
	for (i = 0; i < M_SIZE; i ++) {
		for (j = 0; j < M_SIZE; j ++) {
			A[i*M_SIZE + j] = (rand() % 256) - 128;
			B[i*M_SIZE + j] = (rand() % 256) - 128;
		}
	}
	matrixTranspose(B);

	fprintf(f,"Using an array sized %d by %d, times were calculated for %d iterations of Matrix Multiply methods in seconds.\n\n", M_SIZE, M_SIZE, ITERATIONS);
	
	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		matrixMultiply(A,B,C_sol);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	fprintf(f,"Sequential version finished,\t\t\ttime %f\n", seq_time);

	
	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		UNROLLmatrixMultiply(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"Sequential unrolled version finished,\t\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"Sequential unrolled version failed\n");
	}


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		quad_UNROLLmatrixMultiply(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"Sequential quad unrolled version finished,\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"Sequential quad unrolled version failed\n");
	}
	

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMDmatrixMultiply(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"SIMD version finished,\t\t\t\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"SIMD version failed\n");
	}
	


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMDmatrixMultiplyNew(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"SIMD new version finished,\t\t\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"SIMD new version failed\n");
	}



	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMDmatrixMultiplyQuad(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"SIMD quad version finished,\t\t\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"SIMD quad version failed\n");
	}


	matrixTranspose(B);

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		REGmatrixMultiply(A,B,C);
	}
	end_time = get_sec();
	seq_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"Regular sequential version finished,\t\ttime %f\n",seq_time);
	}
	else {
		fprintf(f,"Regular sequential version failed\n");
	}


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMDmatrixMultiplyGS(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"Regular sequential unrolled version finished,\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"Regular sequential unrolled version failed\n");
	}


	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		SIMDmatrixMultiplyGS(A,B,C);
	}
	end_time = get_sec();
	new_time = end_time - start_time;
	if (check_mat(C_sol,C)) {
		fprintf(f,"SIMD gather version finished,\t\t\ttime %f - x%f\n",new_time,seq_time/new_time);
	}
	else {
		fprintf(f,"SIMD gather version failed\n");
	}

	fclose(f);
	return 0;
}
