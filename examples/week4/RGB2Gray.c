
/*
 *  gcc -msse4.2 -g -O2 RGB2Gray.c -o RGB2Gray
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


void serial_rgb2gray(float* ra, float* ga, float* ba, float* gray) {
    int i;
    for(i = 0; i < N; i++) {
        gray[i] = 0.3f * ra[i] + 0.59f * ga[i] + 0.11f * ba[i];
    }
}

void sse_rgb2gray(float* ra, float* ga, float* ba, float* gray) {
	__m128 rCoEf = _mm_set1_ps(0.3f);
	__m128 gCoEf = _mm_set1_ps(0.59f);
	__m128 bCoEf = _mm_set1_ps(0.11f);
	__m128 R,G,B,Gr;

	int i;
	for (i = 0; i < N; i += 4) {
		R = _mm_loadu_ps(ra + i);
		G = _mm_loadu_ps(ga + i);
		B = _mm_loadu_ps(ba + i);
		Gr = _mm_add_ps(_mm_add_ps(_mm_mul_ps(rCoEf,R),_mm_mul_ps(gCoEf,G)),_mm_mul_ps(bCoEf,B));
		_mm_storeu_ps(gray + i,Gr);
	}
	for(i; i < N; i++) {
        gray[i] = 0.3f * ra[i] + 0.59f * ga[i] + 0.11f * ba[i];
    }
}

float r[N];
float g[N];
float b[N];
float gray_seq[N];
float gray_simd[N];

#define ITERATIONS 1000000
int main (int argc, char* argv[])
{
    int i;
    double start_time, end_time;
    int incorrect = 0;
    for(i = 0; i < N; i++) {
        r[i] = random() % 256;
        g[i] = random() % 256;
        b[i] = random() % 256;
    }
    printf("Convert %d pixels RGB to gray.\n", N);

    start_time = get_sec();
    for(i = 0; i < ITERATIONS; i++) { serial_rgb2gray(r, g, b, gray_seq);}
    end_time = get_sec();
    printf("Sequential version finished, time %f\n", (end_time-start_time));

    start_time = get_sec();
    for(i = 0; i < ITERATIONS; i++) { sse_rgb2gray(r, g, b, gray_simd);}
    end_time = get_sec();


    //Check result
    for(i = 0; i < N; i++) {
      if(abs(gray_simd[i] - gray_seq[i]) > 0.001f) {
        printf("Diff:%f @ %d, seq=%f\n",gray_simd[i] - gray_seq[i], i, gray_seq[i]);
        incorrect = 1;
      }
    }
    if(incorrect){
        printf("SIMD version's result is wrong!!!\n");
    }else{
      printf("SIMD version finished, time %f\n", (end_time-start_time));
    }


    return 0;
}

