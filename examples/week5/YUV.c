/*
 *
 * YUV equations taken from
 *
 * http://www.cse.msu.edu/~cbowen/docs/yuvtorgb.html
 *
 */

#define VECTOR_SIZE 4096

short int R[VECTOR_SIZE];
short int G[VECTOR_SIZE];
short int B[VECTOR_SIZE];
short int Y[VECTOR_SIZE];
short int U[VECTOR_SIZE];
short int V[VECTOR_SIZE];

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

int main() {

	int i;

	for (i=0; i<1000; i++) {
		convertRGBtoYUV();
		convertYUVtoRGB();
	}
	return 0;
}
