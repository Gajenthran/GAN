/*!
 * \file matrix.h
 * \brief Fichier header de matrix.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _MATRIX_H_
#define _MATRIX_H_

// transposée pour le premier argument du produit scalaire
#define LEFT_TRANSPOSE 1
// transposée pour le second argument du produit scalaire
#define RIGHT_TRANSPOSE 2

typedef struct matrix matrix_t;
/* Structure représentant une matrice */
struct matrix {
  int rows; // nombre de lignes
  int cols; // nombre de colonnes
  double* data; // valeurs
};

matrix_t* mat_zinit(int, int);
matrix_t* mat_dot(matrix_t*, matrix_t*);
matrix_t* mat_sigmoid(matrix_t*);
matrix_t* mat_dsigmoid(matrix_t*);
matrix_t* mat_dlrelu(matrix_t*, double);
matrix_t* mat_dtanh(matrix_t*);
void mat_dot_(matrix_t*, matrix_t*, matrix_t*, int);
void mat_lrelu_(matrix_t*, matrix_t*, double);
void mat_tanh_(matrix_t*, matrix_t*);
void mat_sigmoid_(matrix_t*, matrix_t*);
void mat_copy_(matrix_t*, matrix_t*, int);
void mat_sub_(matrix_t*, matrix_t*, matrix_t*);
void mat_sum_axis0_(matrix_t*, matrix_t*);
void mat_sum_(matrix_t*, matrix_t*, matrix_t*);
void mat_mul_(matrix_t*, matrix_t*, matrix_t*);
void mat_mul_scalar(matrix_t*, double);
void mat_ce_(matrix_t*, matrix_t*, matrix_t*);
void mat_log_(matrix_t*, matrix_t*);
void mat_sum_z_act(matrix_t*, matrix_t*, matrix_t*, matrix_t*);
double mat_mean(matrix_t*);
void mat_print_param(matrix_t*);
void mat_print(matrix_t*);
void mat_free(matrix_t*);

#endif