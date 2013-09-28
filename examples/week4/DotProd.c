
/*
 *  gcc -msse4.2 -g -O2 DotProd.c -o DotProd
 * */

#include <stdio.h>
#include <sys/time.h>
#include <nmmintrin.h> //SSE4.2 header


#define N (1000)

static inline double get_sec()
{
  struct timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}


float seq_dot_prod(float* A, float* B) {
	float total = 0;
    int i;
    for(i = 0; i < N; i++) {
        total += A[i]*B[i];
    }

	return total;
}

float sse_dot_prod(float* A, float* B) {
	float sum = 0;
	float temp[4];
	__m128 v1,v2;
	__m128 total = _mm_set1_ps(0.0f);

	int i;
	for (i = 0; i < N; i += 4) {
		v1 = _mm_loadu_ps(A + i);
		v2 = _mm_loadu_ps(B + i);
		total = _mm_add_ps(total,_mm_mul_ps(v1,v2));
	}
	_mm_storeu_ps(temp,total);
	sum = temp[0] + temp[1] + temp[2] + temp[3];

	for(i; i < N; i++) {
        sum += A[i]*B[i];
    }

	return sum;
}

float v1[N];
float v2[N];
float seq;
float simd;

#define ITERATIONS 1000000
int main (int argc, char* argv[])
{
    int i;
    double start_time, end_time;
    int incorrect = 0;
    for(i = 0; i < N; i++) {
        v1[i] = (rand() % 16) - 8;
        v2[i] = (rand() % 16) - 8;
    }
    printf("Dot Product of 2 array with %d elements.\n", N);

    start_time = get_sec();
    for(i = 0; i < ITERATIONS; i++) { seq = seq_dot_prod(v1,v2);}
    end_time = get_sec();
    printf("Sequential version finished, time %f\n", (end_time-start_time));

    start_time = get_sec();
    for(i = 0; i < ITERATIONS; i++) { simd = sse_dot_prod(v1,v2);}
    end_time = get_sec();

    //Check result
    if(seq != simd) {
        incorrect = 1;
    }

    if(incorrect){
        printf("SIMD version's result is wrong!!!\n");
    }else{
      printf("SIMD version finished, time %f\n", (end_time-start_time));
    }


    return 0;
}

