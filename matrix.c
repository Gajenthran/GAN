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
#define DSIGMOID(y) ((y) * (1 - (y)))

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

void mat_adam_opt(matrix_t *w, matrix_t *ms, matrix_t *vs, int it, double beta1, double beta2, double alpha)
{
    const double eps = 1e-8;
    int r, c;

    matrix_t *ms_corr = mat_zinit(ms->rows, ms->cols);
    matrix_t *vs_corr = mat_zinit(ms->rows, ms->cols);

    for (r = 0; r < ms->rows; r++)
        for (c = 0; c < ms->cols; c++)
            ms_corr->data[r][c] = ms->data[r][c] / (1 - pow(beta1, it));

    for (r = 0; r < ms->rows; r++)
        for (c = 0; c < ms->cols; c++)
            vs_corr->data[r][c] = vs->data[r][c] / (1 - pow(beta2, it));

    for (r = 0; r < w->rows; r++)
        for (c = 0; c < w->cols; c++)
            w->data[r][c] = w->data[r][c] - alpha * (ms_corr->data[r][c] / (sqrt(vs_corr->data[r][c]) + eps));
}

matrix_t *reshape(matrix_t *a, matrix_t *src)
{
    if (src->rows * src->cols != a->cols * a->rows)
    {
        printf("Erreur reshape...\n");
        exit(0);
    }

    int r, c;
    double *array = (double *)malloc(src->rows * src->cols * sizeof(*array));
    assert(array);

    matrix_t *res = mat_init(src->rows, src->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            array[r * a->cols + c] = a->data[r][c];

    for (r = 0; r < src->rows; r++)
        for (c = 0; c < src->cols; c++)
            res->data[r][c] = array[r * src->rows + c];

    return res;
}

/** \brief Initialise une matrice en mettant
 * des valeurs aléatoires ([0, 1]).
 *
 * \param rows nombre de lignes
 * \param cols nombre de colonnes
 */
matrix_t *mat_init(int rows, int cols)
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
        {
            mat->data[r][c] = ((float)rand() / (float)(RAND_MAX));
        }
    }
    mat->rows = rows;
    mat->cols = cols;

    return mat;
}

/** \brief Transforme un vecteur 1D en une matrice.
 *
 * \param array vecteur 1D
 * \param size taille du vecteur
 */
matrix_t *array_to_mat(double *array, int size)
{
    matrix_t *mat = mat_init(1, size);

    int i;
    for (i = 0; i < size; i++)
        mat->data[0][i] = array[i];

    return mat;
}

/** \brief Addition matricielle.
 *
 * \param a matrice a
 * \param b matrice b
 */
matrix_t *mat_sum(matrix_t *a, matrix_t *b)
{

    int r, c;
    matrix_t *mat = mat_init(a->rows, a->cols);

    if (a->rows == b->rows && a->cols == b->cols)
    {
        for (r = 0; r < a->rows; r++)
            for (c = 0; c < a->cols; c++)
                mat->data[r][c] = a->data[r][c] + b->data[r][c];
    }
    else if (a->cols == b->cols && b->rows == 1)
    {
        for (r = 0; r < a->rows; r++)
            for (c = 0; c < a->cols; c++)
                mat->data[r][c] = a->data[r][c] + b->data[0][c];
    }
    else
    {
        printf("Error: bad matrix structures while sum\n");
        exit(0);
    }
    return mat;
}

/** \brief Addition matricielle.
 *
 * \param a matrice a
 * \param b matrice b
 */
void mat_sum_src(matrix_t *src, matrix_t *a, matrix_t *b)
{
    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            src->data[r][c] = a->data[r][c] + b->data[0][c];
}

/** \brief Soustraction matricielle.
 *
 * \param a matrice
 * \param val valeur pour soustraire la matrice
 */
matrix_t *mat_sub_val(matrix_t *a, int val)
{
    matrix_t *res = mat_init(a->rows, a->cols);
    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = c == val ? 1.0 - a->data[r][c] : -a->data[r][c];

    return res;
}

matrix_t *mat_linear(matrix_t *a)
{
    return a;
}

#define LRELU(x, alpha) (MAX((x), (x * alpha)))
matrix_t *mat_lrelu(matrix_t *a, double alpha)
{
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
        {
            res->data[r][c] = LRELU(a->data[r][c], alpha);
        }

    return res;
}

// #define TANH(x) (2 / (1 + (exp(-2 * (x)))))
matrix_t *mat_tanh(matrix_t *a)
{
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = tanh(a->data[r][c]);

    return res;
}

#define DTANH(x) ((1.0) - (pow((tanh((x))), 2)))
matrix_t *mat_dtanh(matrix_t *a)
{
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = DTANH(a->data[r][c]);

    return res;
}

matrix_t *mat_exp_beta(matrix_t *ms, matrix_t *grad, double beta)
{
    matrix_t *m = mat_zinit(ms->rows, ms->cols);

    int r, c;

    for (r = 0; r < m->rows; r++)
        for (c = 0; c < m->cols; c++)
            m->data[r][c] = beta * ms->data[r][c] + (1 - beta) * grad->data[r][c];
    return m;
}

matrix_t *mat_exp_sqr_beta(matrix_t *vs, matrix_t *grad, double beta)
{
    matrix_t *m = mat_zinit(vs->rows, vs->cols);

    int r, c;

    for (r = 0; r < m->rows; r++)
        for (c = 0; c < m->cols; c++)
            m->data[r][c] = beta * vs->data[r][c] + (1 - beta) * pow(grad->data[r][c], 2);

    return m;
}

#define DRELU(x) ((x) > 0 ? 1 : 0)
matrix_t *mat_drelu(matrix_t *a)
{
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = DRELU(a->data[r][c]);

    return res;
}

/** \brief Soustraction matricielle.
 *
 * \param a matrice
 * \param val valeur pour soustraire la matrice
 */
matrix_t *mat_loss(matrix_t *output, matrix_t *target)
{
    matrix_t *res = mat_init(output->rows, output->cols);
    int r, c;
    for (r = 0; r < output->rows; r++)
        for (c = 0; c < output->cols; c++)
            res->data[r][c] = target->data[r][c] - output->data[r][c];

    return res;
}

matrix_t *mat_sub(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
    {
        printf("%d = %d || %d = %d\n", a->rows, b->rows, a->cols, b->cols);
        printf("Error: bad matrix structures while sub\n");
        exit(0);
    }

    matrix_t *res = mat_init(a->rows, a->cols);
    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = a->data[r][c] - b->data[r][c];

    return res;
}

/** \brief Produit matriciel.
 *
 * \param a matrice a
 * \param b matrice b
 */
matrix_t *mat_mul(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
    {
        printf("Error: bad matrix structures while mul\n");
        exit(0);
    }

    matrix_t *res = mat_init(a->rows, b->cols);

    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < b->cols; c++)
            res->data[r][c] = a->data[r][c] * b->data[r][c];
    return res;
}

/** \brief Produit matriciel entre une valeur et une matrice.
 *
 * \param a matrice
 * \param val valeur pour multiplier la matrice
 */
void mat_mul_scalar(matrix_t *a, double val)
{
    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            a->data[r][c] *= val;
}

/** \brief Transposée d'une matrice.
 *
 * \param a matrice
 */
matrix_t *mat_transpose(matrix_t *a)
{
    matrix_t *res = mat_init(a->cols, a->rows);

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
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = SIGMOID(a->data[r][c]);

    return res;
}

/** \brief Applique la fonction dérivée de 
 *la sigmoïde sur la matrice.
 *
 * \param a matrice
 */
matrix_t *mat_dsigmoid(matrix_t *a)
{
    int r, c;
    matrix_t *res = mat_init(a->rows, a->cols);

    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            res->data[r][c] = DSIGMOID(a->data[r][c]);

    return res;
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
        printf("Erreur: mat_dot\n");
        exit(1);
    }

    matrix_t *res = mat_init(a->rows, b->cols);

    int r, c, k;
    for (r = 0; r < a->rows; r++)
    {
        for (c = 0; c < b->cols; c++)
        {
            res->data[r][c] = 0;
            for (k = 0; k < b->rows; k++)
            {
                res->data[r][c] += a->data[r][k] * b->data[k][c];
            }
        }
    }

    return res;
}

/** \brief Redimensionner la matrice en transformant
 * celle-ci sous forme de matrice c*1 où c correspond
 * au nombre de colonnes de la matrice de départ.
 *
 * \param a matrice
 */
matrix_t *mat_reshape_col(matrix_t *a)
{
    if (a->rows != 1)
    {
        printf("Erreur: mat_reshape_col\n");
        exit(1);
    }

    matrix_t *res = mat_init(a->cols, 1);
    int c;
    for (c = 0; c < a->cols; c++)
    {
        res->data[c][0] = a->data[0][c];
    }

    return res;
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

double mat_get_value(matrix_t *mat, int r, int c)
{
    return mat->data[r][c];
}

#define DBCE(x, y) (((x) - (y)) / ((x) * (1 - (x))))
matrix_t *mat_dbce(matrix_t *pred, matrix_t *labels)
{
    matrix_t *loss = mat_init(pred->rows, pred->cols);
    int r, c;
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            loss->data[r][c] = DBCE(pred->data[r][c], labels->data[r][c]);

    return loss;
}

#define BCE(x, y) ((y)*log(x) + (1 - (y)) * log(1 - (x)))
double mat_bce(matrix_t *pred, matrix_t *labels)
{
    double sum = 0.0;
    int r, c;

    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            sum += BCE(pred->data[r][c], labels->data[r][c]);

    return -sum / (pred->cols * pred->rows);
}

double mat_bce_loss(matrix_t *pred, matrix_t *labels, int normalize)
{
    int r, c, n;

    n = normalize ? pred->rows * pred->cols : 1;

    double loss = 0.0;
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            loss += labels->data[r][c] * log(pred->data[r][c]);

    return -loss;
}

#define CE(x, y) ((-log((y))) - (log(1 - (x))))
double mat_ce_loss(matrix_t *pred, matrix_t *labels)
{
    int r, c;

    double loss = 0.0;
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            loss += CE(pred->data[r][c], labels->data[r][c]);

    return loss / labels->rows;
}

matrix_t *mat_ce(matrix_t *pred, matrix_t *labels)
{
    int r, c;

    matrix_t *a = mat_zinit(pred->rows, pred->cols);
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            a->data[r][c] = CE(pred->data[r][c], labels->data[r][c]);

    return a;
}

double mat_log_loss(matrix_t *pred)
{
    int r, c;

    double loss = 0.0;
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            loss += -log(pred->data[r][c]);

    return loss / pred->rows;
}

matrix_t *mat_log(matrix_t *pred)
{
    int r, c;

    matrix_t *a = mat_zinit(pred->rows, pred->cols);
    for (r = 0; r < pred->rows; r++)
        for (c = 0; c < pred->cols; c++)
            a->data[r][c] = -log(pred->data[r][c]);

    return a;
}

matrix_t *mat_dsoftmax(matrix_t *input)
{
    int r = 0.0, c;
    double sum = 0.0;
    matrix_t *res = mat_init(input->rows, input->cols);

    for (c = 0; c < input->cols; c++)
    {
        res->data[r][c] = exp(input->data[r][c]) / sum;
    }

    return res;
}
matrix_t *mat_softmax(matrix_t *input)
{
    int r, c;
    double sum;
    matrix_t *res = mat_init(input->rows, input->cols);

    sum = 0.0;
    for (r = 0; r < input->rows; r++)
        for (c = 0; c < input->cols; c++)
            sum += exp(input->data[r][c]);

    for (r = 0; r < input->rows; r++)
        for (c = 0; c < input->cols; c++)
        {
            printf("%.2f ->", input->data[r][c]);
            res->data[r][c] = exp(input->data[r][c]) / sum;
        }

    printf("\n");
    return res;
}

double mat_mse(matrix_t *y_pred, matrix_t *y)
{
    int r, c;
    double sum = 0.0;
    for (r = 0; r < y->rows; r++)
        for (c = 0; c < y->cols; c++)
            sum += pow(y_pred->data[r][c] - y->data[r][c], 2.0);

    return sum / (y->rows * y->cols);
}

matrix_t *mat_dmse(matrix_t *y_pred, matrix_t *y)
{
    int r, c;
    matrix_t *res = mat_zinit(y->rows, y->cols);
    for (r = 0; r < y->rows; r++)
        for (c = 0; c < y->cols; c++)
            res->data[r][c] = y_pred->data[r][c] - y->data[r][c];

    return res;
}

matrix_t *mat_copy(matrix_t *src, int i_min, int m_sz)
{
    matrix_t *mat = mat_zinit(m_sz, src->cols);
    int r, c;
    for (r = 0; r < m_sz; r++)
        for (c = 0; c < src->cols; c++)
            mat->data[r][c] = src->data[i_min + r][c];

    return mat;
}

matrix_t *mat_sum_axis0(matrix_t *src)
{
    matrix_t *a = mat_zinit(1, src->cols);

    int r, c;
    double sum = 0.0;
    for (r = 0; r < src->cols; r++)
    {
        for (c = 0; c < src->rows; c++)
            sum += src->data[c][r];
        a->data[0][r] = sum;
        sum = 0.0;
    }
    return a;
}

matrix_t *mat_vinit(matrix_t *src, double value)
{
    matrix_t *a = mat_zinit(src->rows, src->cols);

    int r, c;
    for (r = 0; r < a->rows; r++)
        for (c = 0; c < a->cols; c++)
            a->data[r][c] = value;

    return a;
}

matrix_t *mat_dlrelu(matrix_t *x, double alpha)
{
    matrix_t *dx = mat_vinit(x, 1);

    int r, c;
    for (r = 0; r < dx->rows; r++)
        for (c = 0; c < dx->cols; c++)
            if (x->data[r][c] < 0)
                dx->data[r][c] = alpha;

    return dx;
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