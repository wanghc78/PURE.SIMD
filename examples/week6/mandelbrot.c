#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nmmintrin.h>

static inline double get_sec()
{
  struct timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

/* 
                Scalar version of mandelbrot 
*/
static int mandel(float c_re, float c_im, int count) {
  float z_re = c_re, z_im = c_im;
  int cci = 0;
  for (cci = 0; cci < count; ++cci) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }
  return cci;
}

void mandelbrot_serial(float x0, float y0, float x1, float y1, int width,
  int height, int maxIterations, int output[]) {
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;
  int i, j;
  for (j = 0; j < height; j++) {
    for (i = 0; i < width; ++i) {
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}


/* 
                SIMD version of mandelbrot 
*/
static __m128i SIMDmandel(__m128 c_re, __m128 c_im, int count) {
	__m128 z_re = c_re, z_im = c_im;
	__m128 cons2 = _mm_set1_ps(2.f);
	__m128 cons4 = _mm_set1_ps(4.f);
	__m128 mask = _mm_set1_ps(0xffffffff);

	int cci = 0;
	int check;
	__m128i ret = _mm_set1_epi32(cci);
	for (cci = 0; cci < count; ++cci) {
		__m128i temp = _mm_set1_epi32(cci);

		ret = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(ret),_mm_castsi128_ps(temp),mask));
		mask = _mm_cmple_ps(_mm_add_ps(_mm_mul_ps(z_re,z_re),_mm_mul_ps(z_im,z_im)),cons4);
		check = _mm_movemask_ps(mask);
		if (!check) {
			break;
		}

		__m128 new_re = _mm_sub_ps(_mm_mul_ps(z_re,z_re),_mm_mul_ps(z_im,z_im));
		__m128 new_im = _mm_mul_ps(_mm_mul_ps(cons2,z_re),z_im);
		z_re = _mm_add_ps(c_re,new_re);
		z_im = _mm_add_ps(c_im,new_im);
	}
	if (cci == count) {
		__m128i temp = _mm_set1_epi32(cci);
		ret = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(ret),_mm_castsi128_ps(temp),mask));
	}
	return ret;
}

void mandelbrot_SIMD(float x0, float y0, float x1, float y1, int width, int height, int maxIterations, int output[]) {
	__m128i out;
	__m128 Si, Sj;
	__m128 dx = _mm_set1_ps((x1 - x0) / width);
	__m128 dy = _mm_set1_ps((y1 - y0) / height);
	__m128 Sx0 = _mm_set1_ps(x0);
	__m128 Sy0 = _mm_set1_ps(y0);
	
	int i, j;
	for (j = 0; j < height; j ++) {
		Sj = _mm_set1_ps(j);
		for (i = 0; i < width; i += 4) {
			Si = _mm_set_ps((float) (i + 3),(float) (i + 2),(float) (i + 1),(float) i);
			__m128 x = _mm_add_ps(Sx0,_mm_mul_ps(Si,dx));
			__m128 y = _mm_add_ps(Sy0,_mm_mul_ps(Sj,dy));

			int index = (j * width + i);
			out = SIMDmandel(x, y, maxIterations);
			_mm_storeu_si128((__m128i *) (output + index),out);
		}
	}
}


/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    int i,j;
    for (i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

#define ITERATIONS 3

int main() {
  unsigned int width = 768;
  unsigned int height = 512;

  //unsigned int* width = 1024;
  //unsigned int* height = 1024;

  float x0 = -2;
  float x1 = 1;
  float y0 = -1;
  float y1 = 1;

  int maxIterations = 10;
  int buf[width * height];
  //int bufS[width * height];

  //
  // Compute the image using the scalar and generic intrinsics implementations; report the minimum
  // time of three runs.
  //
  double start_time, end_time;
  int i;

  start_time = get_sec();
  for(i = 0; i < ITERATIONS; i++) {
    mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
  }
  end_time = get_sec();
  double shnr = (end_time-start_time);
  printf("Sequential version finished, time %f\n", (end_time-start_time));

  writePPM(buf, width, height, "mandelbrot-serial.ppm");

  start_time = get_sec();
  for(i = 0; i < ITERATIONS; i++) {
    mandelbrot_SIMD(x0, y0, x1, y1, width, height, maxIterations, buf);
  }
  end_time = get_sec();
  printf("SIMD version finished, time %f - x%f\n", (end_time-start_time),shnr/(end_time-start_time));
  
  writePPM(buf, width, height, "mandelbrot-SIMD.ppm");

  return 0;
}
