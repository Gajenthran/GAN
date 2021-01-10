/*!
 * \file matrix.h
 * \brief Fichier header de matrix.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _MATRIX_H_
#define _MATRIX_H_

/* Structure représentant une matrice */
typedef struct matrix matrix_t;
struct matrix
{
    int rows;      // nombre de lignes
    int cols;      // nombre de colonnes
    double **data; // données
};

matrix_t *mat_init(int, int);
matrix_t *mat_zinit(int, int);
matrix_t *mat_sum(matrix_t *, matrix_t *);
matrix_t *mat_sub(matrix_t *, matrix_t *);
matrix_t *mat_loss(matrix_t *, matrix_t *);
matrix_t *mat_mul(matrix_t *, matrix_t *);
void mat_mul_scalar(matrix_t *, double);
matrix_t *mat_sigmoid(matrix_t *);
matrix_t *mat_dsigmoid(matrix_t *);
matrix_t *mat_transpose(matrix_t *);
matrix_t *mat_dot(matrix_t *, matrix_t *);
matrix_t *mat_reshape_col(matrix_t *a);
matrix_t *array_to_mat(double *, int);
void mat_free(matrix_t *);
double mat_get_value(matrix_t *, int, int);
matrix_t *mat_dbce(matrix_t *pred, matrix_t *labels);
matrix_t *mat_lrelu(matrix_t *, double);
matrix_t *mat_drelu(matrix_t *);
matrix_t *mat_exp_beta(matrix_t *, matrix_t *, double);
matrix_t *mat_exp_sqr_beta(matrix_t *, matrix_t *, double);
void mat_adam_opt(matrix_t *, matrix_t *, matrix_t *, int, double, double, double);
double mat_bce_loss(matrix_t *, matrix_t *, int);
matrix_t *mat_linear(matrix_t *a);
matrix_t *mat_softmax(matrix_t *input);
double mat_bce(matrix_t *, matrix_t *);
double mat_mse(matrix_t *y_pred, matrix_t *y);
matrix_t *mat_dmse(matrix_t *y_pred, matrix_t *y);
matrix_t *mat_tanh(matrix_t *);
matrix_t *mat_copy(matrix_t *src, int i_min, int m_sz);
double mat_ce_loss(matrix_t *pred, matrix_t *labels);
matrix_t *mat_sum_axis0(matrix_t *);
matrix_t *mat_sub_val(matrix_t *a, int val);
matrix_t *mat_dtanh(matrix_t *a);
void mat_sum_src(matrix_t *src, matrix_t *a, matrix_t *b);
void mat_print_param(matrix_t *mat);
double mat_log_loss(matrix_t *pred);
double mat_mean(matrix_t *x);
void mat_ce_(matrix_t *, matrix_t *, matrix_t *);
void mat_log_(matrix_t *, matrix_t *);
void mat_print(matrix_t *);
void mat_sum_(matrix_t *src, matrix_t *a, matrix_t *b);
void mat_dot_(matrix_t *src, matrix_t *a, matrix_t *b);
void mat_lrelu_(matrix_t *src, matrix_t *a, double alpha);
void mat_tanh_(matrix_t *src, matrix_t *a);
void mat_sigmoid_(matrix_t *src, matrix_t *a);
void mat_copy_(matrix_t *src, matrix_t *a, int i_min);
void mat_sub_(matrix_t *src, matrix_t *a, matrix_t *b);
double mat_error(matrix_t *a, matrix_t *b);
void mat_mul_(matrix_t *src, matrix_t *a, matrix_t *b);
void mat_sum_axis0_(matrix_t *src, matrix_t *a);
void mat_free(matrix_t *mat);
double mat_sum_values(matrix_t *a);
void mat_dsigmoid_(matrix_t *, matrix_t *);
matrix_t *mat_dlrelu(matrix_t *x, double alpha);
matrix_t *mat_dsigmoid(matrix_t *a);
#endif