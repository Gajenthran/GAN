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

/** \brief Initialise une matrice en mettant
 * les valeurs à 0.
 *
 * \param rows nombre de lignes
 * \param cols nombre de colonnes
 */
matrix_t *mat_zinit(int rows, int cols)
{
  matrix_t *mat = (matrix_t *)malloc(sizeof(*mat));
  assert(mat);

  mat->data = (double **)malloc(rows * sizeof(*mat->data));
  assert(mat->data);

  int r, c;
  for (r = 0; r < rows; r++)
  {
    mat->data[r] = (double *)malloc(cols * sizeof(*mat->data[r]));
    assert(mat->data[r]);

    for (c = 0; c < cols; c++)
      mat->data[r][c] = 0.0;
  }

  mat->rows = rows;
  mat->cols = cols;

  return mat;
}

matrix_t *mat_fill(matrix_t *src, double value)
{
  matrix_t *a = mat_zinit(src->rows, src->cols);

  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      a->data[r][c] = value;

  return a;
}

void mat_sum_(matrix_t *src, matrix_t *a, matrix_t *b)
{
  int r, c;
  if (a->rows == b->rows && a->cols == b->cols)
  {
    for (r = 0; r < a->rows; r++)
      for (c = 0; c < a->cols; c++)
        src->data[r][c] = a->data[r][c] + b->data[r][c];
  }
  else if (a->cols == b->cols && b->rows == 1)
  {
    for (r = 0; r < a->rows; r++)
      for (c = 0; c < a->cols; c++)
        src->data[r][c] = a->data[r][c] + b->data[0][c];
  }
  else
  {
    fprintf(stderr, "Error: bad matrix structures while sum. \n");
    exit(1);
  }
}

void mat_lrelu_(matrix_t *src, matrix_t *a, double alpha)
{
  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      src->data[r][c] = LRELU(a->data[r][c], alpha);
}

void mat_tanh_(matrix_t *src, matrix_t *a)
{
  int r, c;

  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      src->data[r][c] = tanh(a->data[r][c]);
}

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
      src->data[r][c] = a->data[r][c] - b->data[r][c];
}

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
      src->data[r][c] = a->data[r][c] * b->data[r][c];
}

/** \brief Transposée d'une matrice.
 *
 * \param a matrice
 */
matrix_t *mat_transpose(matrix_t *a)
{
  matrix_t *res = mat_zinit(a->cols, a->rows);

  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      res->data[c][r] = a->data[r][c];

  return res;
}

/** \brief Applique la fonction sigmoïde sur
 * la matrice.
 *
 * \param a matrice
 */
matrix_t *mat_sigmoid(matrix_t *a)
{
  matrix_t *res = mat_zinit(a->rows, a->cols);

  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      res->data[r][c] = SIGMOID(a->data[r][c]);

  return res;
}

matrix_t *mat_dsigmoid(matrix_t *a)
{
  matrix_t *res = mat_zinit(a->rows, a->cols);

  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      res->data[r][c] = DSIGMOID(a->data[r][c]);

  return res;
}

matrix_t *mat_dlrelu(matrix_t *x, double alpha)
{
  matrix_t *dx = mat_fill(x, 1);

  int r, c;
  for (r = 0; r < dx->rows; r++)
    for (c = 0; c < dx->cols; c++)
      if (x->data[r][c] < 0)
        dx->data[r][c] = alpha;

  return dx;
}

matrix_t *mat_dtanh(matrix_t *a)
{
  matrix_t *res = mat_zinit(a->rows, a->cols);

  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      res->data[r][c] = DTANH(a->data[r][c]);

  return res;
}

void mat_sigmoid_(matrix_t *src, matrix_t *a)
{
  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      src->data[r][c] = SIGMOID(a->data[r][c]);
}

void mat_dot_(matrix_t *src, matrix_t *a, matrix_t *b)
{
  if (src->rows != a->rows ||
      src->cols != b->cols ||
      a->cols != b->rows)
  {
    fprintf(stderr, "Error: bad matrix structures while dot. \n");
    exit(1);
  }

  int r, c, k;
  for (r = 0; r < a->rows; r++)
  {
    for (c = 0; c < b->cols; c++)
    {
      src->data[r][c] = 0;
      for (k = 0; k < b->rows; k++)
        src->data[r][c] += a->data[r][k] * b->data[k][c];
    }
  }
}

/** \brief Libère la mémoire de la matrice.
 *
 * \param mat matrice
 */
void mat_free(matrix_t *mat)
{
  if (mat)
  {
    int r;
    for (r = 0; r < mat->rows; r++)
      free(mat->data[r]);
    free(mat->data);
    free(mat);
  }
}

/** \brief Affiche la matrice.
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
      printf("%.3f, ", mat->data[r][c]);
    printf("],\n");
  }
  printf("]\n\n");
}

void mat_print_param(matrix_t *mat)
{
  printf("rows: %d, cols:%d\n", mat->rows, mat->cols);
}

void mat_ce_(matrix_t *src, matrix_t *pred, matrix_t *labels)
{
  int r, c;
  for (r = 0; r < pred->rows; r++)
    for (c = 0; c < pred->cols; c++)
      src->data[r][c] = CE(pred->data[r][c], labels->data[r][c]);
}

void mat_log_(matrix_t *src, matrix_t *pred)
{
  int r, c;
  for (r = 0; r < pred->rows; r++)
    for (c = 0; c < pred->cols; c++)
      src->data[r][c] = -log(pred->data[r][c]);
}

void mat_copy_(matrix_t *src, matrix_t *a, int i_min)
{
  int r, c;
  for (r = 0; r < src->rows; r++)
    for (c = 0; c < src->cols; c++)
      src->data[r][c] = a->data[i_min + r][c];
}

void mat_sum_axis0_(matrix_t *src, matrix_t *a)
{
  int r, c;
  double sum = 0.0;
  for (r = 0; r < a->cols; r++)
  {
    for (c = 0; c < a->rows; c++)
      sum += a->data[c][r];
    src->data[0][r] = sum;
    sum = 0.0;
  }
}

double mat_mean(matrix_t *x)
{
  double sum = 0.0;
  int r, c;
  for (r = 0; r < x->rows; r++)
    for (c = 0; c < x->cols; c++)
      sum += x->data[r][c];
  return sum / x->rows;
}

/** \brief Produit scalaire de la matrice.
 *
 * \param a matrice a
 * \param b matrice b
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
        res->data[r][c] += a->data[r][k] * b->data[k][c];

  return res;
}

void mat_sum_z_act(matrix_t *z, matrix_t *act, matrix_t *w, matrix_t *b)
{
  matrix_t *dot = mat_dot(act, w);
  mat_sum_(z, dot, b);
  mat_free(dot);
}

void mat_der_(matrix_t *src, matrix_t *a, matrix_t *b, unsigned char first)
{
  matrix_t *t = first ? a : b;
  matrix_t *tp_val = mat_transpose(t);

  if (first)
    mat_mul_(src, tp_val, b);
  else
    mat_mul_(src, a, tp_val);

  mat_free(tp_val);
}

void mat_mul_scalar(matrix_t *a, double val)
{
  int r, c;
  for (r = 0; r < a->rows; r++)
    for (c = 0; c < a->cols; c++)
      a->data[r][c] *= val;
}

double mat_sum_values(matrix_t *src)
{
  int i, j;
  double sum = 0.0;
  for (i = 0; i < src->rows; i++)
    for (j = 0; j < src->cols; j++)
      sum += src->data[i][j];
  return sum;
}