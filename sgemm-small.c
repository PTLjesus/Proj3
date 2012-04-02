//Name: Yijia Zang
//Login: cs61c-fp

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void square_sgemm_manager (int n, float* A, float* B, float* C);
void transpose( int n, int blocksize, float *dst, float *src );
void edge_handling( int n, float *dst, float* src );
void depadding(int n, int size, float* dst, float* src);

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * index = i + j*N (i is row, j is column)
 * On exit, A and B maintain their input values. */  

void square_sgemm (int n, float* A, float* B, float* C)
{  
	int size = n + (4 - n%4); 
	int blockSize1 = 32;
	int blockSize2 = 16;

	float *betterA = (float*) calloc(size*size, sizeof(float));
	float *betterB = (float*) calloc(size*size, sizeof(float));
	float *betterC = (float*) calloc(size*size, sizeof(float));
	float *transposeA = (float*) calloc(n*n, sizeof(float));
	float *betterTransposeA = (float*) calloc(size*size, sizeof(float));

	if(betterA == NULL || betterB == NULL || betterC == NULL || betterTransposeA == NULL || transposeA == NULL)
	{
		printf("Memory could not be allocated!\n");
		exit(101);
	}

	if(n%4 != 0)
	{
		transpose(n,32, transposeA, A);

		edge_handling(n, betterTransposeA, transposeA);
		edge_handling(n, betterB, B);
		edge_handling(n, betterC, C);	

		square_sgemm_manager(size, betterTransposeA, betterB, betterC);
		depadding(n, size, C, betterC);		
	}

	else
	{
		transpose(n, 32, transposeA, A);
		square_sgemm_manager(n, transposeA, B, C);
	}

	free(betterA);
	free(betterB);
	free(betterC);
	free(betterTransposeA);
	free(transposeA);

	betterA = NULL;
	betterB = NULL;
	betterC = NULL;
	betterTransposeA = NULL;
	transposeA = NULL;
}


void square_sgemm_manager (int n, float* A, float* B, float* C)
{  
	__m128 a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3;

		for( int i = 0; i < n; i+=4 ) //right by 4
		{
			for(int j = 0; j < n; j += 4) //down by 4
			{
				//  compute a 4 by 4 matrix
				c0 = _mm_loadu_ps( C + (i + 0)*n + j ); /* c0 = (C[0],C[1],C[2],C[3]) column0 */
				c1 = _mm_loadu_ps( C + (i + 1)*n + j ); /* c1 = (C[n],C[n+1],C[n+2],C[n+3]) column1 */
				c2 = _mm_loadu_ps( C + (i + 2)*n + j ); /* c1 = (C[2n],C[2n+1],C[2n+2],C[2n+3]) column2 */
				c3 = _mm_loadu_ps( C + (i + 3)*n + j ); /* c2 = (C[3n],C[3n+1],C[3n+2],C[3n+3]) column3 */

	
				for( int k = 0; k < n; k+=4 ) //iterate through the columns
				{
					a0  = _mm_loadu_ps( A + (0+j)*n + k ); // a = (a[i*n+0],a[i*n+1],a[i*n+2],a[i*n+3])
					a1  = _mm_loadu_ps( A + (1+j)*n + k );
					a2  = _mm_loadu_ps( A + (2+j)*n + k );
					a3  = _mm_loadu_ps( A + (3+j)*n + k );
	
					b0 = _mm_loadu_ps( B + (i + 0)*n + k ); // b0 = (b[i],b[i],b[i],b[i])
					b1 = _mm_loadu_ps( B + (i + 1)*n + k ); // b0 = (b[2n+i],b[2n+i],b[2n+i],b[2n+i])
					b2 = _mm_loadu_ps( B + (i + 2)*n + k ); // b0 = (b[3n+i],b[3n+i],b[3n+i],b[3n+i])
					b3 = _mm_loadu_ps( B + (i + 3)*n + k ); // b0 = (b[4n+i],b[4n+i],b[4n+i],b[4n+i])
	
					// multiply and add 
					c0 = _mm_add_ps (c0, _mm_hadd_ps ( _mm_hadd_ps(_mm_mul_ps(b0, a0), _mm_mul_ps(b0, a1)), _mm_hadd_ps(_mm_mul_ps(b0, a2), _mm_mul_ps(b0, a3) ) ) ); 
					c1 = _mm_add_ps (c1, _mm_hadd_ps ( _mm_hadd_ps(_mm_mul_ps(b1, a0), _mm_mul_ps(b1, a1)), _mm_hadd_ps(_mm_mul_ps(b1, a2), _mm_mul_ps(b1, a3) ) ) );
					c2 = _mm_add_ps (c2, _mm_hadd_ps ( _mm_hadd_ps(_mm_mul_ps(b2, a0), _mm_mul_ps(b2, a1)), _mm_hadd_ps(_mm_mul_ps(b2, a2), _mm_mul_ps(b2, a3) ) ) );
					c3 = _mm_add_ps (c3, _mm_hadd_ps ( _mm_hadd_ps(_mm_mul_ps(b3, a0), _mm_mul_ps(b3, a1)), _mm_hadd_ps(_mm_mul_ps(b3, a2), _mm_mul_ps(b3, a3) ) ) );
				}

				/* store the result back to the C array */
				_mm_storeu_ps( C + (i + 0)*n + j, c0 ); /* (C[0],C[1],C[2],C[3]) = c0 */
				_mm_storeu_ps( C + (i + 1)*n + j, c1 ); /* (C[n],C[n+1],C[n+2],C[n+3]) = c1 */
				_mm_storeu_ps( C + (i + 2)*n + j, c2 ); /* (C[2n],C[2n+1],C[2n+2],C[2n+3]) = c2 */
				_mm_storeu_ps( C + (i + 3)*n + j, c3 ); /* (C[3n],C[3n+1],C[3n+2],C[3n+3]) = c3 */
			}
		}
}


void transpose( int n, int blocksize, float *dst, float *src ) 
{
    int i, j, ii, jj;
	
    for( i = 0; i < n; i += blocksize )		
        for( j = 0; j < n; j += blocksize )	
			for(ii = i; ii < i + blocksize && ii < n; ii++)
				for(jj = j; jj < j +blocksize && jj < n; jj++)
					dst[ii+jj*n] = src[jj+ii*n];
}

void edge_handling(int n, float *dst, float* src)
{
	int i, j, new_size = n+(4-n%4);

	for(i = 0; i < n; i ++)
		for(j = 0; j < n; j++)
			dst[j+i*new_size] = src[j+i*n];
}


void depadding(int n, int size, float* dst, float* src)
{
	int i, j;

	for(i = 0; i < n; i ++)
		for(j = 0; j < n; j++)
			dst[i + j*n] = src[i + j*size];
}