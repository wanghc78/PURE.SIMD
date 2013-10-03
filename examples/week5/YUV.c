/*
 *
 * YUV equations taken from
 *
 * http://www.cse.msu.edu/~cbowen/docs/yuvtorgb.html
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <nmmintrin.h>

#define VECTOR_SIZE 4096

short int R[VECTOR_SIZE];
short int G[VECTOR_SIZE];
short int B[VECTOR_SIZE];
short int Y[VECTOR_SIZE];
short int U[VECTOR_SIZE];
short int V[VECTOR_SIZE];

static inline double get_sec()
{
  struct timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

void convertRGBtoYUV() {

	int i;

	for (i=0; i<VECTOR_SIZE; i++) {
		Y[i] = (R[i]*77 + G[i]*150 + B[i]*29);
		U[i] = (R[i]*-43 + G[i]*-85 + B[i]*128 + 32767);
		V[i] = (R[i]*128 + G[i]*-107 + B[i]*-21 + 32767);
		Y[i] = Y[i] + 256;
		U[i] = U[i] + 256;
		V[i] = V[i] + 256;
		Y[i] = Y[i] >> 8;
		U[i] = U[i] >> 8;
		V[i] = V[i] >> 8;
	}
}

void convertYUVtoRGB() {

	int i;
	for (i=0; i<VECTOR_SIZE; i++) {
		Y[i] = Y[i] << 8;
		R[i] = (Y[i]+(360*(V[i]-128)));
		G[i] = (Y[i]-(88*(U[i]-128) - (184*(V[i]-128))));
		B[i] = (Y[i]+(455*(U[i]-128)));
		R[i] = R[i] + 256;
		G[i] = G[i] + 256;
		B[i] = B[i] + 256;
		R[i] = R[i] >> 8;
		G[i] = G[i] >> 8;
		B[i] = B[i] >> 8;
	}
}

void AconvertRGBtoYUV() {
	__m128i r,g,b,y,u,v;
	__m128i rCo1 = _mm_set1_epi16((short)77);
	__m128i gCo1 = _mm_set1_epi16((short)150);
	__m128i bCo1 = _mm_set1_epi16((short)29);
	__m128i rCo2 = _mm_set1_epi16((short)-43);
	__m128i gCo2 = _mm_set1_epi16((short)-85);
	__m128i bCo2 = _mm_set1_epi16((short)128);
	__m128i rCo3 = _mm_set1_epi16((short)128);
	__m128i gCo3 = _mm_set1_epi16((short)-107);
	__m128i bCo3 = _mm_set1_epi16((short)-21);
	__m128i cons1 = _mm_set1_epi16((short)32767);
	__m128i cons2 = _mm_set1_epi16((short)256);

	int remainder = VECTOR_SIZE % 8;
	int i;
	for (i = 0; i < VECTOR_SIZE - remainder; i += 8) {
		r = _mm_loadu_si128((__m128i *) (R + i));
		g = _mm_loadu_si128((__m128i *) (G + i));
		b = _mm_loadu_si128((__m128i *) (B + i));
		y = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(r,rCo1),_mm_mullo_epi16(g,gCo1)),_mm_mullo_epi16(b,bCo1));
		u = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(r,rCo2),_mm_mullo_epi16(g,gCo2)),_mm_mullo_epi16(b,bCo2)),cons1);
		v = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(r,rCo3),_mm_mullo_epi16(g,gCo3)),_mm_mullo_epi16(b,bCo3)),cons1);
		y = _mm_add_epi16(y,cons2);
		u = _mm_add_epi16(u,cons2);
		v = _mm_add_epi16(v,cons2);
		y = _mm_srai_epi16(y,8);
		u = _mm_srai_epi16(u,8);
		v = _mm_srai_epi16(v,8);
		_mm_storeu_si128((__m128i *) (Y + i),y);
		_mm_storeu_si128((__m128i *) (U + i),u);
		_mm_storeu_si128((__m128i *) (V + i),v);
	}
	for (i; i < VECTOR_SIZE; i ++) {
		Y[i] = (R[i]*77 + G[i]*150 + B[i]*29);
		U[i] = (R[i]*-43 + G[i]*-85 + B[i]*128 + 32767);
		V[i] = (R[i]*128 + G[i]*-107 + B[i]*-21 + 32767);
		Y[i] = Y[i] + 256;
		U[i] = U[i] + 256;
		V[i] = V[i] + 256;
		Y[i] = Y[i] >> 8;
		U[i] = U[i] >> 8;
		V[i] = V[i] >> 8;
	}
}

void AconvertYUVtoRGB() {
	__m128i r,g,b,y,u,v;
	__m128i vCo1 = _mm_set1_epi16((short)360);
	__m128i uCo1 = _mm_set1_epi16((short)88);
	__m128i vCo2 = _mm_set1_epi16((short)184);
	__m128i uCo2 = _mm_set1_epi16((short)455);
	__m128i cons1 = _mm_set1_epi16((short)-128);
	__m128i cons2 = _mm_set1_epi16((short)256);

	int remainder = VECTOR_SIZE % 8;
	int i;
	for (i = 0; i < VECTOR_SIZE - remainder; i += 8) {
		y = _mm_loadu_si128((__m128i *) (Y + i));
		u = _mm_loadu_si128((__m128i *) (U + i));
		v = _mm_loadu_si128((__m128i *) (V + i));
		y = _mm_slli_epi16(y,8);
		r = _mm_add_epi16(y,_mm_mullo_epi16(vCo1,_mm_add_epi16(v,cons1)));
		g = _mm_sub_epi16(y,_mm_sub_epi16(_mm_mullo_epi16(uCo1,_mm_add_epi16(u,cons1)),_mm_mullo_epi16(vCo2,_mm_add_epi16(v,cons1))));
		b = _mm_add_epi16(y,_mm_mullo_epi16(uCo2,_mm_add_epi16(u,cons1)));
		r = _mm_add_epi16(r,cons2);
		b = _mm_add_epi16(b,cons2);
		g = _mm_add_epi16(g,cons2);
		r = _mm_srai_epi16(r,8);
		b = _mm_srai_epi16(b,8);
		g = _mm_srai_epi16(g,8);
		_mm_storeu_si128((__m128i *) (Y + i),y);
		_mm_storeu_si128((__m128i *) (R + i),r);
		_mm_storeu_si128((__m128i *) (G + i),g);
		_mm_storeu_si128((__m128i *) (B + i),b);
	}
	for (i; i < VECTOR_SIZE; i ++) {
		Y[i] = Y[i] << 8;
		R[i] = (Y[i]+(360*(V[i]-128)));
		G[i] = (Y[i]-(88*(U[i]-128) - (184*(V[i]-128))));
		B[i] = (Y[i]+(455*(U[i]-128)));
		R[i] = R[i] + 256;
		G[i] = G[i] + 256;
		B[i] = B[i] + 256;
		R[i] = R[i] >> 8;
		G[i] = G[i] >> 8;
		B[i] = B[i] >> 8;
	}
}

#define ITERATIONS 1000

int main() {
	int i;
	double start_time;
	double end_time;

	for (i = 0; i < VECTOR_SIZE; i ++) {
		R[i] = (short) (rand() % 256);
		G[i] = (short) (rand() % 256);
		B[i] = (short) (rand() % 256);
	}

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		convertRGBtoYUV();
		convertYUVtoRGB();
	}
	end_time = get_sec();
	printf("Sequential version finished, time %f\n", (end_time-start_time));

	start_time = get_sec();
	for(i = 0; i < ITERATIONS; i++) {
		AconvertRGBtoYUV();
		AconvertYUVtoRGB();
	}
	end_time = get_sec();
	printf("SIMD version finished, time %f\n", (end_time-start_time));

	return 0;
}
