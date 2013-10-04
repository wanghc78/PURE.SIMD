#include <stdio.h>
#include <stdlib.h>

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



int main() {
  unsigned int width = 768;
  unsigned int height = 512;

  //unsigned int width = 1024;
  //unsigned int height = 1024;

  float x0 = -2;
  float x1 = 1;
  float y0 = -1;
  float y1 = 1;

  int maxIterations = 10;
  int buf[width * height];

  //
  // Compute the image using the scalar and generic intrinsics implementations; report the minimum
  // time of three runs.
  //
  mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
  writePPM(buf, width, height, "mandelbrot-serial.ppm");

  return 0;
}
