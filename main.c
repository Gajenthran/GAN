/*!
 * \file ann.c
 * \brief Fichier principale concernant 
 * l'application de SOM
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include "matrix.h"

#define BATCH_SZ 64

typedef struct cfg_t cfg_t;
typedef struct gan_t gan_t;

struct cfg_t
{
  int num_batches;
  int batch_size;
  int train_size;
  matrix_t *x_train;
  matrix_t *y_train;
};

struct gan_t
{
  int epochs;
  int input_layer_sz_g;
  int hidden_layer_sz_g;
  int hidden_layer_sz_d;
  double lr;
  double dr;

  matrix_t *w0_g;
  matrix_t *b0_g;

  matrix_t *w1_g;
  matrix_t *b1_g;

  matrix_t *w0_d;
  matrix_t *b0_d;

  matrix_t *w1_d;
  matrix_t *b1_d;

  matrix_t *z0_g;
  matrix_t *a0_g;
  matrix_t *z1_g;
  matrix_t *x_fake;

  matrix_t *z0_d;
  matrix_t *a0_d;
  matrix_t *z1_d;
  matrix_t *a1_d;

  matrix_t *z1_d_real, *a1_d_real;
  matrix_t *z1_d_fake, *a1_d_fake;
};

void usage(char *msg)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

double rand_gen()
{
  // return a uniformly distributed random value
  return ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
}
double normal_rand()
{
  // return a normally distributed random value
  double v1 = rand_gen();
  double v2 = rand_gen();
  return cos(2 * 3.14 * v2) * sqrt(-2. * log(v1));
}

cfg_t *preprocess_data(void)
{
  int i, j = 0, size = 0;
  for (i = 0; i < NUM_TRAIN; i++)
    if (train_label_char[i][0] == 3)
      size++;

  int num_batches = size / BATCH_SZ;
  size = num_batches * BATCH_SZ;
  matrix_t *x_train = mat_zinit(size, SIZE);
  matrix_t *y_train = mat_zinit(size, 1);

  // Take only label with digit 3
  for (i = 0; i < NUM_TRAIN; i++)
  {
    if (train_label_char[i][0] == 1)
    {
      x_train->data[j] = train_image[i];
      y_train->data[j][0] = (double)((int)train_label_char[i][0]);
      j++;
    }

    if (j == size)
      break;
  }

  // TODO: Shuffle

  cfg_t *cfg = (cfg_t *)malloc(sizeof(*cfg));
  assert(cfg);

  cfg->x_train = x_train;
  cfg->y_train = y_train;
  cfg->train_size = size;
  cfg->batch_size = BATCH_SZ;
  cfg->num_batches = num_batches;

  return cfg;
}

gan_t *init_gan(void)
{
  int i, j;

  int input_layer_sz_g = 100;
  int hidden_layer_sz_g = 128;
  int hidden_layer_sz_d = 128;

  // w0_g
  matrix_t *w0_g = mat_init(input_layer_sz_g, hidden_layer_sz_g);
  for (i = 0; i < input_layer_sz_g; i++)
    for (j = 0; j < hidden_layer_sz_g; j++)
      w0_g->data[i][j] = normal_rand() * sqrt(2.0 / input_layer_sz_g);

  // b0_g
  matrix_t *b0_g = mat_zinit(1, hidden_layer_sz_g);

  // w1_g
  matrix_t *w1_g = mat_init(hidden_layer_sz_g, SIZE);
  for (i = 0; i < hidden_layer_sz_g; i++)
    for (j = 0; j < SIZE; j++)
      w1_g->data[i][j] = normal_rand() * sqrt(2.0 / hidden_layer_sz_g);

  // b1_g
  matrix_t *b1_g = mat_zinit(1, SIZE);

  // w0_d
  matrix_t *w0_d = mat_init(SIZE, hidden_layer_sz_d);
  for (i = 0; i < SIZE; i++)
    for (j = 0; j < hidden_layer_sz_d; j++)
      w0_d->data[i][j] = normal_rand() * (sqrt(2.0 / SIZE));

  // b0_d
  matrix_t *b0_d = mat_zinit(1, hidden_layer_sz_d);

  // w1_d
  matrix_t *w1_d = mat_init(hidden_layer_sz_d, 1);
  for (i = 0; i < hidden_layer_sz_d; i++)
    w1_d->data[i][0] = normal_rand() * sqrt(2.0 / hidden_layer_sz_d);

  // b1_d
  matrix_t *b1_d = mat_zinit(1, 1);

  gan_t *gan = (gan_t *)malloc(1 * sizeof(*gan));
  assert(gan);

  gan->hidden_layer_sz_d = hidden_layer_sz_d;
  gan->hidden_layer_sz_g = hidden_layer_sz_g;
  gan->input_layer_sz_g = input_layer_sz_g;
  gan->w0_d = w0_d;
  gan->w1_d = w1_d;
  gan->w0_g = w0_g;
  gan->w1_g = w1_g;
  gan->b0_d = b0_d;
  gan->b1_d = b1_d;
  gan->b0_g = b0_g;
  gan->b1_g = b1_g;
  gan->lr = 1e-3;
  gan->dr = 1e-4;
  gan->epochs = 100;

  return gan;
}

void forward_g(gan_t *gan, matrix_t *z)
{
  gan->z0_g = mat_sum(mat_dot(z, gan->w0_g), gan->b0_g);
  gan->a0_g = mat_lrelu(gan->z0_g, 0);

  gan->z1_g = mat_sum(mat_dot(gan->a0_g, gan->w1_g), gan->b1_g);
  gan->x_fake = mat_tanh(gan->z1_g);
}

void forward_d(gan_t *gan, matrix_t *x, int real)
{
  gan->z0_d = mat_sum(mat_dot(x, gan->w0_d), gan->b0_d);
  gan->a0_d = mat_lrelu(gan->z0_d, 1e-2);

  if (real)
  {
    gan->z1_d_real = mat_sum(mat_dot(gan->a0_d, gan->w1_d), gan->b1_d);
    gan->a1_d_real = mat_sigmoid(gan->z1_d_real);
  }
  else
  {
    gan->z1_d_fake = mat_sum(mat_dot(gan->a0_d, gan->w1_d), gan->b1_d);
    gan->a1_d_fake = mat_sigmoid(gan->z1_d_fake);
  }
}

void backward_d(gan_t *gan, matrix_t *x_real)
{
  int r, c;

  matrix_t *da1_real = mat_zinit(gan->a1_d_real->rows, gan->a1_d_real->cols);
  for (r = 0; r < da1_real->rows; r++)
    for (c = 0; c < da1_real->cols; c++)
      da1_real->data[r][c] = -1.0 / (gan->a1_d_real->data[r][c] + 1e-8);

  matrix_t *dz1_real = mat_mul(da1_real, mat_dsigmoid(mat_sigmoid(gan->z1_d_real)));
  matrix_t *dw1_real = mat_dot(mat_transpose(gan->a0_d), dz1_real);
  matrix_t *db1_real = mat_sum_axis0(dz1_real);

  matrix_t *da0_real = mat_dot(dz1_real, mat_transpose(gan->w1_d));
  matrix_t *dz0_real = mat_mul(da0_real, mat_dlrelu(gan->z0_d, 1e-2));

  matrix_t *dw0_real = mat_dot(mat_transpose(x_real), dz0_real);
  matrix_t *db0_real = mat_sum_axis0(dz0_real);

  matrix_t *da1_fake = mat_zinit(gan->a1_d_fake->rows, gan->a1_d_fake->cols);
  for (r = 0; r < da1_real->rows; r++)
    for (c = 0; c < da1_real->cols; c++)
      da1_fake->data[r][c] = 1.0 / (1.0 - gan->a1_d_fake->data[r][c] + 1e-8);

  matrix_t *dz1_fake = mat_mul(da1_fake, mat_dsigmoid(mat_sigmoid(gan->z1_d_fake)));
  matrix_t *dw1_fake = mat_dot(mat_transpose(gan->a0_d), dz1_fake);
  matrix_t *db1_fake = mat_sum_axis0(dz1_fake);

  matrix_t *da0_fake = mat_dot(dz1_fake, mat_transpose(gan->w1_d));
  matrix_t *dz0_fake = mat_mul(da0_fake, mat_dlrelu(gan->z0_d, 0));
  matrix_t *dw0_fake = mat_dot(mat_transpose(gan->x_fake), dz0_fake);
  matrix_t *db0_fake = mat_sum_axis0(dz0_fake);

  matrix_t *dw1 = mat_sum(dw1_real, dw1_fake);
  matrix_t *db1 = mat_sum(db1_real, db1_fake);

  matrix_t *dw0 = mat_sum(dw0_real, dw0_fake);
  matrix_t *db0 = mat_sum(db0_real, db0_fake);

  mat_mul_scalar(dw0, gan->lr);
  gan->w0_d = mat_sub(gan->w0_d, dw0);
  mat_mul_scalar(db0, gan->lr);
  gan->b0_d = mat_sub(gan->b0_d, db0);
  mat_mul_scalar(dw1, gan->lr);
  gan->w1_d = mat_sub(gan->w1_d, dw1);
  mat_mul_scalar(db1, gan->lr);
  gan->b1_d = mat_sub(gan->b1_d, db1);

  free(dz1_fake);
  free(dw1_fake);
  free(db1_fake);

  free(da0_fake);
  free(dz0_fake);
  free(dw0_fake);
  free(db0_fake);

  free(dw1);
  free(db1);

  free(dw0);
  free(db0);
}

void backward_g(gan_t *gan, matrix_t *z)
{
  int r, c;

  matrix_t *da1_d = mat_zinit(gan->a1_d_fake->rows, gan->a1_d_fake->cols);
  for (r = 0; r < da1_d->rows; r++)
    for (c = 0; c < da1_d->cols; c++)
      da1_d->data[r][c] = -1.0 / (gan->a1_d_fake->data[r][c] + 1e-8);

  matrix_t *dz1_d = mat_mul(da1_d, mat_dsigmoid(mat_sigmoid(gan->z1_d_fake)));
  matrix_t *da0_d = mat_dot(dz1_d, mat_transpose(gan->w1_d));
  matrix_t *dz0_d = mat_mul(da0_d, mat_dlrelu(gan->z0_d, 1e-2));
  matrix_t *dx_d = mat_dot(dz0_d, mat_transpose(gan->w0_d));

  matrix_t *dz1_g = mat_mul(dx_d, mat_dtanh(gan->z1_g));
  matrix_t *dw1_g = mat_dot(mat_transpose(gan->a0_g), dz1_g);
  matrix_t *db1_g = mat_sum_axis0(dz1_g);

  matrix_t *da0_g = mat_dot(dz1_g, mat_transpose(gan->w1_g));
  matrix_t *dz0_g = mat_mul(da0_g, mat_dlrelu(gan->z0_g, 0));
  matrix_t *dw0_g = mat_dot(mat_transpose(z), dz0_g);
  matrix_t *db0_g = mat_sum_axis0(dz0_g);

  mat_mul_scalar(dw0_g, gan->lr);
  gan->w0_g = mat_sub(gan->w0_g, dw0_g);
  mat_mul_scalar(db0_g, gan->lr);
  gan->b0_g = mat_sub(gan->b0_g, db0_g);
  mat_mul_scalar(dw1_g, gan->lr);
  gan->w1_g = mat_sub(gan->w1_g, dw1_g);
  mat_mul_scalar(db1_g, gan->lr);
  gan->b1_g = mat_sub(gan->b1_g, db1_g);

  free(dz1_d);
  free(da0_d);
  free(dz0_d);
  free(dx_d);

  free(dz1_g);
  free(dw1_g);
  free(db1_g);

  free(da0_g);
  free(dz0_g);
  free(dw0_g);
  free(db0_g);
}

void train_gan(cfg_t *cfg, gan_t *gan)
{
  int i, j, bs, il, i_min;
  double losses_d, losses_g, avg_d_fake, avg_d_real;
  losses_d = losses_g = avg_d_fake = avg_d_real = 0.0;

  matrix_t *z = mat_init(cfg->batch_size, gan->input_layer_sz_g);

  for (i = 0; i < gan->epochs; i++)
  {
    for (j = 0; j < cfg->num_batches; j++)
    {
      for (bs = 0; bs < cfg->batch_size; bs++)
        for (il = 0; il < gan->input_layer_sz_g; il++)
          z->data[bs][il] = normal_rand();

      i_min = i * cfg->batch_size;
      matrix_t *x_real = mat_copy(cfg->x_train, i_min, cfg->batch_size);

      forward_g(gan, z);
      forward_d(gan, x_real, 1);
      forward_d(gan, gan->x_fake, 0);

      backward_d(gan, x_real);
      backward_g(gan, z);

      losses_d = mat_mean(mat_ce(gan->a1_d_fake, gan->a1_d_real));
      losses_g = mat_mean(mat_log(gan->a1_d_fake));

      free(x_real);
    }
    if (i % 5 == 0)
    {
      printf("epoch:  %d\n", i);
      printf("lr:     %f\n", gan->lr);
      printf("loss_g: %.3f\n", losses_g);
      printf("loss_d: %.3f\n", losses_d);
      printf("avg_df: %.3f\n", mat_mean(gan->a1_d_fake));
      printf("avg_dr: %.3f\n\n", mat_mean(gan->a1_d_real));
      save_mnist_pgm_mat(gan->x_fake);
    }

    losses_d = losses_g = 0.0;
    gan->lr = gan->lr * (1.0 / (1.0 + gan->dr * i));
  }
}

int main(void)
{
  srand(time(NULL));

  load_mnist();

  cfg_t *cfg = preprocess_data();

  gan_t *gan = init_gan();
  train_gan(cfg, gan);
  return 0;
}