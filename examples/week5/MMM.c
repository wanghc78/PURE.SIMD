#define M_SIZE 256

float A[M_SIZE*M_SIZE];
float B[M_SIZE*M_SIZE];
float C[M_SIZE*M_SIZE];

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

int main() {
	int i;

	for (i=0; i<10; i++) {
		matrixTranspose(B);
		matrixMultiply(A,B,C);
	}
	return 0;
}
