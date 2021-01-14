/*!
 * \file matrix.c
 * \brief Fichier comprenant les fonctionnalités
 * pour gérer une matrice.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

#define MAX(a, b) \
	({ __typeof__ (a) _a = (a); \
			 __typeof__ (b) _b = (b); \
		 _a > _b ? _a : _b; })

#define SIGMOID(x) (1 / (1 + (exp(-(x)))))
#define LRELU(x, alpha) (MAX((x), (x * alpha)))
#define DSIGMOID(y) ((y) * (1 - (y)))
#define DRELU(x) ((x) > 0 ? 1 : 0)
#define DTANH(x) ((1.0) - (pow((tanh((x))), 2)))
#define CE(x, y) ((-log((y))) - (log(1 - (x))))

/** \brief Initialiser une matrice en mettant
 * les valeurs à 0.
 *
 * \param rows nombre de lignes
 * \param cols nombre de colonnes
 * \return structure matrix
 */
matrix_t *mat_zinit(int rows, int cols)
{
	matrix_t *mat = (matrix_t *)malloc(sizeof(*mat));
	assert(mat);

	mat->data = (double *)calloc(rows * cols, sizeof(*mat->data));
	assert(mat->data);

	mat->rows = rows;
	mat->cols = cols;
	return mat;
}

/** \brief Initialiser une matrice en remplissant
 * les valeurs avec la valeur passée en paramètre.
 *
 * \param src matrice
 * \param value valeur
 * \return structure matrix
 */
matrix_t *mat_fill(matrix_t *src, double value)
{
	matrix_t *a = mat_zinit(src->rows, src->cols);

	int i;
	for (i = 0; i < a->rows * a->cols; i++)
		a->data[i] = value;

	return a;
}

/** \brief Somme de deux matrices (a + b).
 *
 * \param src matrice source
 * \param a matrice a
 * \param b matrice b
 */
void mat_sum_(matrix_t *src, matrix_t *a, matrix_t *b)
{
	int r, c, i;
	if (a->rows == b->rows && a->cols == b->cols)
	{
		for (r = 0; r < a->rows; r++)
			for (c = 0; c < a->cols; c++)
			{
				i = r * a->cols + c;
				src->data[i] = a->data[i] + b->data[i];
			}
	}
	else if (a->cols == b->cols && b->rows == 1)
	{
		for (r = 0; r < a->rows; r++)
			for (c = 0; c < a->cols; c++)
			{
				i = r * a->cols + c;
				src->data[i] = a->data[i] + b->data[0 * a->cols + c];
			}
	}
	else
	{
		fprintf(stderr, "Error: bad matrix structures while sum. \n");
		exit(1);
	}
}

/** \brief Appliquer la fonction RELU sur la matrice a.
 *
 * \param src matrice source
 * \param a matrice a
 * \param alpha coefficient d'apprentissage
 */
void mat_lrelu_(matrix_t *src, matrix_t *a, double alpha)
{
	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			src->data[r * a->cols + c] = LRELU(a->data[r * a->cols + c], alpha);
}

/** \brief Appliquer la fonction tanh sur la matrice a.
 *
 * \param src matrice source
 * \param a matrice a
 */
void mat_tanh_(matrix_t *src, matrix_t *a)
{
	int r, c;

	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			src->data[r * a->cols + c] = tanh(a->data[r * a->cols + c]);
}

/** \brief Soustraction de deux matrices (a - b).
 *
 * \param src matrice source
 * \param a matrice a
 * \param b matrice b
 */
void mat_sub_(matrix_t *src, matrix_t *a, matrix_t *b)
{
	if (a->rows != b->rows || a->cols != b->cols)
	{
		fprintf(stderr, "Error: bad matrix structures while sub. \n");
		exit(1);
	}

	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			src->data[r * a->cols + c] = a->data[r * a->cols + c] - b->data[r * a->cols + c];
}

/** \brief Multiplication de deux matrices (a * b).
 *
 * \param src matrice source
 * \param a matrice a
 * \param b matrice b
 */
void mat_mul_(matrix_t *src, matrix_t *a, matrix_t *b)
{
	if (src->rows != a->rows ||
			src->cols != b->cols ||
			a->rows != b->rows || a->cols != b->cols)
	{
		fprintf(stderr, "Error: bad matrix structures while mul\n");
		exit(1);
	}

	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < b->cols; c++)
			src->data[r * a->cols + c] = a->data[r * a->cols + c] * b->data[r * a->cols + c];
}

/** \brief Appliquer la fonction sigmoïde sur la matrice a.
 *
 * \param a matrice a
 * \return matrice a avec l'application de sigmoïde
 */
matrix_t *mat_sigmoid(matrix_t *a)
{
	matrix_t *res = mat_zinit(a->rows, a->cols);

	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			res->data[r * a->cols + c] = SIGMOID(a->data[r * a->cols + c]);

	return res;
}

/** \brief Appliquer la dérivée de sigmoïde sur la matrice a.
 *
 * \param a matrice a
 * \return matrice a avec la dérivée de sigmoïde
 */
matrix_t *mat_dsigmoid(matrix_t *a)
{
	matrix_t *res = mat_zinit(a->rows, a->cols);

	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			res->data[r * a->cols + c] = DSIGMOID(a->data[r * a->cols + c]);

	return res;
}

/** \brief Appliquer la dérivée de RELU sur la matrice a.
 *
 * \param a matrice a
 * \param alpha coefficient d'apprentissage
 * \return matrice a avec la dérivée de RELU
 */
matrix_t *mat_dlrelu(matrix_t *a, double alpha)
{
	matrix_t *res = mat_fill(a, 1);

	int r, c;
	for (r = 0; r < res->rows; r++)
		for (c = 0; c < res->cols; c++)
			if (a->data[r * res->cols + c] < 0)
				res->data[r * res->cols + c] = alpha;

	return res;
}

/** \brief Appliquer la dérivée de tanh sur la matrice a.
 *
 * \param a matrice a
 * \return matrice a avec la dérivée de tanh
 */
matrix_t *mat_dtanh(matrix_t *a)
{
	matrix_t *res = mat_zinit(a->rows, a->cols);

	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			res->data[r * a->cols + c] = DTANH(a->data[r * a->cols + c]);

	return res;
}

/** \brief Appliquer la fonction sigmoïde sur la matrice a.
 *
 * \param src matrice source
 * \param a matrice a
 */
void mat_sigmoid_(matrix_t *src, matrix_t *a)
{
	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			src->data[r * a->cols + c] = SIGMOID(a->data[r * a->cols + c]);
}

/** \brief Appliquer le produit scalaire sur la matrice a et b.
 *
 * \param src matrice source
 * \param a matrice a
 * \param b matrice b
 * \param transpose appliquer la transposée sur a ou b
 */
void mat_dot_(matrix_t *src, matrix_t *a, matrix_t *b, int transpose)
{
	int rows = transpose == LEFT_TRANSPOSE ? a->cols : a->rows;
	int cols = transpose == LEFT_TRANSPOSE ? b->cols : b->rows;
	int com = transpose == LEFT_TRANSPOSE ? b->rows : a->cols;

	if (src->rows != rows ||
			src->cols != cols)
	{
		fprintf(stderr, "Error: bad matrix structures while dot. \n");
		exit(1);
	}

	int r, c, k;
	double tmp = 0.0;

	for (r = 0; r < rows; r++)
	{
		for (c = 0; c < cols; c++)
		{
			tmp = 0.0;
			for (k = 0; k < com; k++)
			{
				if (transpose == LEFT_TRANSPOSE)
					tmp += a->data[k * a->cols + r] * b->data[k * b->cols + c];
				else
					tmp += a->data[r * a->cols + k] * b->data[c * b->cols + k];
			}
			src->data[r * src->cols + c] = tmp;
		}
	}
}

/** \brief Libérer la mémoire de la matrice.
 *
 * \param mat matrice
 */
void mat_free(matrix_t *mat)
{
	if (mat)
	{
		free(mat->data);
		free(mat);
		mat = NULL;
	}
}

/** \brief Afficher la matrice.
 *
 * \param mat matrice
 */
void mat_print(matrix_t *mat)
{
	int r, c;
	printf("rows: %d, cols:%d\n", mat->rows, mat->cols);
	printf("[ \n");
	for (r = 0; r < mat->rows; r++)
	{
		printf(" [ ");
		for (c = 0; c < mat->cols; c++)
			printf("%.3f, ", mat->data[r * mat->cols + c]);
		printf("],\n");
	}
	printf("]\n\n");
}

/** \brief Afficher les paramètre de la matrice.
 *
 * \param mat matrice
 */
void mat_print_param(matrix_t *mat)
{
	printf("rows: %d, cols:%d\n", mat->rows, mat->cols);
}

/** \brief Appliquer l'entropie croisée sur la matrice pred.
 *
 * \param src matrice source
 * \param pred matrice de prédiction
 * \param labels matrice avec les labels
 */
void mat_ce_(matrix_t *src, matrix_t *pred, matrix_t *labels)
{
	int r, c;
	for (r = 0; r < pred->rows; r++)
		for (c = 0; c < pred->cols; c++)
			src->data[r * src->cols + c] = CE(pred->data[r * pred->cols + c], labels->data[r * labels->cols + c]);
}

/** \brief Appliquer la fonction de log sur la matrice pred.
 *
 * \param src matrice source
 * \param pred matrice de prédiction
 */
void mat_log_(matrix_t *src, matrix_t *pred)
{
	int r, c;
	for (r = 0; r < pred->rows; r++)
		for (c = 0; c < pred->cols; c++)
			src->data[r * src->cols + c] = -log(pred->data[r * pred->cols + c]);
}

/** \brief Copier la matrice a.
 *
 * \param src matrice source
 * \param a matrice a
 * \param i_min indice à partir duquel il faut copier
 */
void mat_copy_(matrix_t *src, matrix_t *a, int i_min)
{
	int r, c;
	for (r = 0; r < src->rows; r++)
		for (c = 0; c < src->cols; c++)
			src->data[r * src->cols + c] = a->data[(i_min + r) * a->cols + c];
}

/** \brief Somme de toutes les valeurs d'une matrice pour obtenir qu'un
 * seul axe.
 *
 * \param src matrice source
 * \param a matrice a
 */
void mat_sum_axis0_(matrix_t *src, matrix_t *a)
{
	int r, c;
	double sum = 0.0;
	for (r = 0; r < a->cols; r++)
	{
		for (c = 0; c < a->rows; c++)
			sum += a->data[c * a->cols + r];

		src->data[0 * src->cols + r] = sum;
		sum = 0.0;
	}
}

/** \brief Moyenne de toutes les valeurs d'une matrice.
 *
 * \param a matrice a
 * \return valeur moyenne de la matrice a
 */
double mat_mean(matrix_t *a)
{
	double sum = 0.0;
	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			sum += a->data[r * a->cols + c];
	return sum / a->rows;
}

/** \brief Appliquer le produit scalaire sur la matrice a et b.
 *
 * \param src matrice source
 * \param a matrice a
 * \return matrice représentant le produit scalaire de a et b
 */
matrix_t *mat_dot(matrix_t *a, matrix_t *b)
{
	if (a->cols != b->rows)
	{
		fprintf(stderr, "Error: bad matrix structures while dot. \n");
		exit(1);
	}

	matrix_t *res = mat_zinit(a->rows, b->cols);

	int r, c, k;
	for (r = 0; r < a->rows; r++)
		for (k = 0; k < b->rows; k++)
			for (c = 0; c < b->cols; c++)
				res->data[r * res->cols + c] += a->data[r * a->cols + k] * b->data[k * b->cols + c];

	return res;
}

/** \brief Produit scalaire entre une matrice et une valeur.
 *
 * \param a matrice a
 * \param val valeur
 */
void mat_mul_scalar(matrix_t *a, double val)
{
	int r, c;
	for (r = 0; r < a->rows; r++)
		for (c = 0; c < a->cols; c++)
			a->data[r * a->cols + c] *= val;
}

void mat_sum_z_act(matrix_t *z, matrix_t *act, matrix_t *w, matrix_t *b)
{
	matrix_t *dot = mat_dot(act, w);
	mat_sum_(z, dot, b);
	mat_free(dot);
}