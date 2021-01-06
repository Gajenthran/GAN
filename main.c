#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include "matrix.h"

#define BATCH_SZ 64
#define CHOSEN_LABEL 3

typedef struct cfg_t cfg_t;
typedef struct gan_t gan_t;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

enum ACT_E
{
  LRELU = 0,
  SIGMOID,
  TANH
};

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
  int *layers_sz_d;
  int *layers_sz_g;
  int *act_fn_d;
  int *act_fn_g;
  int nb_layers;
  int epochs;
  int input_layer_sz_g;
  int hidden_layer_sz_g;
  int hidden_layer_sz_d;
  double lr;
  double dr;

  matrix_t **w_g;
  matrix_t **b_g;
  matrix_t **z_g;
  matrix_t **a_g;

  matrix_t **w_d;
  matrix_t **b_d;
  matrix_t **z_d_fake;
  matrix_t **z_d_real;
  matrix_t **a_d_fake;
  matrix_t **a_d_real;

  matrix_t **da_g;
  matrix_t **dz_g;
  matrix_t **dw_g;
  matrix_t **db_g;

  matrix_t **da_d;
  matrix_t **dz_d;
  matrix_t **dw_d_real;
  matrix_t **dw_d_fake;
  matrix_t **db_d_real;
  matrix_t **db_d_fake;
};

void usage(char *msg)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

double rand_gen(void)
{
  // return a uniformly distributed random value
  return ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
}
double normal_rand(void)
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
    if (train_label_char[i][0] == CHOSEN_LABEL)
      size++;

  int num_batches = size / BATCH_SZ;
  size = num_batches * BATCH_SZ;
  matrix_t *x_train = mat_zinit(size, SIZE);
  matrix_t *y_train = mat_zinit(size, 1);

  // Take only label with digit 3
  for (i = 0; i < NUM_TRAIN; i++)
  {
    if (train_label_char[i][0] == CHOSEN_LABEL)
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
  int input_layer_sz_g = 100;
  int hidden_layer_sz_g = 128;
  int hidden_layer_sz_d = 128;

  int nb_layers = 3;
  int *layers_sz_d = (int *)malloc(nb_layers * sizeof(*layers_sz_d));
  int *layers_sz_g = (int *)malloc(nb_layers * sizeof(*layers_sz_g));

  int *act_fn_g = (int *)malloc((nb_layers - 1) * sizeof(*act_fn_g));
  int *act_fn_d = (int *)malloc((nb_layers - 1) * sizeof(*act_fn_d));

  act_fn_g[0] = LRELU;
  act_fn_g[1] = TANH;
  act_fn_d[0] = LRELU;
  act_fn_d[1] = SIGMOID;

  layers_sz_d[0] = SIZE;
  layers_sz_d[1] = hidden_layer_sz_d;
  layers_sz_d[2] = 1;

  layers_sz_g[0] = input_layer_sz_g;
  layers_sz_g[1] = hidden_layer_sz_g;
  layers_sz_g[2] = SIZE;

  matrix_t **w_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*w_g));
  matrix_t **b_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*b_g));
  matrix_t **z_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*z_g));
  matrix_t **a_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*a_g));

  matrix_t **w_d = (matrix_t **)malloc((nb_layers - 1) * sizeof(*w_d));
  matrix_t **b_d = (matrix_t **)malloc((nb_layers - 1) * sizeof(*b_d));
  matrix_t **z_d_fake = (matrix_t **)malloc((nb_layers - 1) * sizeof(*z_d_fake));
  matrix_t **z_d_real = (matrix_t **)malloc((nb_layers - 1) * sizeof(*z_d_real));
  matrix_t **a_d_fake = (matrix_t **)malloc((nb_layers - 1) * sizeof(*a_d_fake));
  matrix_t **a_d_real = (matrix_t **)malloc((nb_layers - 1) * sizeof(*a_d_real));

  matrix_t **da_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*a_g));
  matrix_t **dz_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*z_g));
  matrix_t **dw_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*w_g));
  matrix_t **db_g = (matrix_t **)malloc((nb_layers - 1) * sizeof(*b_g));

  matrix_t **da_d = (matrix_t **)malloc((nb_layers - 1) * sizeof(*a_g));
  matrix_t **dz_d = (matrix_t **)malloc((nb_layers - 1) * sizeof(*z_g));
  matrix_t **dw_d_real = (matrix_t **)malloc((nb_layers - 1) * sizeof(*w_g));
  matrix_t **dw_d_fake = (matrix_t **)malloc((nb_layers - 1) * sizeof(*w_g));
  matrix_t **db_d_real = (matrix_t **)malloc((nb_layers - 1) * sizeof(*b_g));
  matrix_t **db_d_fake = (matrix_t **)malloc((nb_layers - 1) * sizeof(*b_g));

  int i, r, c;
  int g_rows, d_rows;
  for (i = 0; i < nb_layers - 1; i++)
  {
    g_rows = (i == 0) ? BATCH_SZ : a_g[i - 1]->rows;
    d_rows = (i == 0) ? BATCH_SZ : a_d_real[i - 1]->rows;

    w_g[i] = mat_init(layers_sz_g[i], layers_sz_g[i + 1]);
    b_g[i] = mat_zinit(1, layers_sz_g[i + 1]);

    w_d[i] = mat_init(layers_sz_d[i], layers_sz_d[i + 1]);
    b_d[i] = mat_zinit(1, layers_sz_d[i + 1]);

    a_g[i] = mat_zinit(g_rows, w_g[i]->cols);
    a_d_fake[i] = mat_zinit(d_rows, w_d[i]->cols);
    a_d_real[i] = mat_zinit(d_rows, w_d[i]->cols);

    z_g[i] = mat_zinit(g_rows, w_g[i]->cols);
    z_d_fake[i] = mat_zinit(d_rows, w_d[i]->cols);
    z_d_real[i] = mat_zinit(d_rows, w_d[i]->cols);

    da_g[i] = mat_zinit(g_rows, w_g[i]->cols);
    dz_g[i] = mat_zinit(g_rows, w_g[i]->cols);
    dw_g[i] = mat_zinit(layers_sz_g[i], layers_sz_g[i + 1]);
    db_g[i] = mat_zinit(1, layers_sz_g[i + 1]);

    da_d[i] = mat_zinit(d_rows, w_d[i]->cols);
    dz_d[i] = mat_zinit(d_rows, w_d[i]->cols);
    dw_d_real[i] = mat_zinit(layers_sz_d[i], layers_sz_d[i + 1]);
    dw_d_fake[i] = mat_zinit(layers_sz_d[i], layers_sz_d[i + 1]);
    db_d_real[i] = mat_zinit(1, layers_sz_d[i + 1]);
    db_d_fake[i] = mat_zinit(1, layers_sz_d[i + 1]);

    for (r = 0; r < layers_sz_g[i]; r++)
      for (c = 0; c < layers_sz_g[i + 1]; c++)
        w_g[i]->data[r][c] = normal_rand() * sqrt(2.0 / layers_sz_g[i]);

    for (r = 0; r < layers_sz_d[i]; r++)
      for (c = 0; c < layers_sz_d[i + 1]; c++)
        w_d[i]->data[r][c] = normal_rand() * sqrt(2.0 / layers_sz_d[i]);
  }

  gan_t *gan = (gan_t *)malloc(1 * sizeof(*gan));
  assert(gan);

  // paramètres
  gan->layers_sz_d = layers_sz_d;
  gan->layers_sz_g = layers_sz_g;
  gan->act_fn_g = act_fn_g;
  gan->act_fn_d = act_fn_d;
  gan->nb_layers = nb_layers;
  gan->hidden_layer_sz_d = hidden_layer_sz_d;
  gan->hidden_layer_sz_g = hidden_layer_sz_g;
  gan->input_layer_sz_g = input_layer_sz_g;
  gan->lr = 1e-3;
  gan->dr = 1e-4;
  gan->epochs = 100;

  // nouveau modèle
  gan->w_g = w_g;
  gan->b_g = b_g;
  gan->w_d = w_d;
  gan->b_d = b_d;
  gan->a_g = a_g;
  gan->z_d_fake = z_d_fake;
  gan->z_d_real = z_d_real;
  gan->a_d_fake = a_d_fake;
  gan->a_d_real = a_d_real;
  gan->z_g = z_g;
  gan->da_g = da_g;
  gan->dz_g = dz_g;
  gan->dw_g = dw_g;
  gan->db_g = db_g;
  gan->da_d = da_d;
  gan->dz_d = dz_d;
  gan->dw_d_real = dw_d_real;
  gan->dw_d_fake = dw_d_fake;
  gan->db_d_real = db_d_real;
  gan->db_d_fake = db_d_fake;

  return gan;
}

void forward_g_(gan_t *gan, matrix_t *z)
{
  int i;
  matrix_t *act = z;
  for (i = 0; i < gan->nb_layers - 1; i++)
  {
    mat_sum_(gan->z_g[i], mat_dot(act, gan->w_g[i]), gan->b_g[i]);

    switch (gan->act_fn_g[i])
    {
    case LRELU:
      mat_lrelu_(gan->a_g[i], gan->z_g[i], 0);
      break;
    case TANH:
      mat_tanh_(gan->a_g[i], gan->z_g[i]);
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }
    act = gan->a_g[i];
  }
}

void forward_d_(gan_t *gan, matrix_t *x, int real)
{
  int i;

  matrix_t **z_d = real ? gan->z_d_real : gan->z_d_fake;
  matrix_t **a_d = real ? gan->a_d_real : gan->a_d_fake;
  matrix_t *act = x;

  for (i = 0; i < gan->nb_layers - 1; i++)
  {
    mat_sum_(z_d[i], mat_dot(act, gan->w_d[i]), gan->b_d[i]);

    switch (gan->act_fn_d[i])
    {
    case LRELU:
      mat_lrelu_(a_d[i], z_d[i], 1e-2);
      break;
    case SIGMOID:
      mat_sigmoid_(a_d[i], z_d[i]);
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }
    act = a_d[i];
  }
}

void backward_d_(gan_t *gan, matrix_t *x_real)
{
  int r, c;

  int i, ri = gan->nb_layers - 2;
  for (r = 0; r < gan->da_d[ri]->rows; r++)
    for (c = 0; c < gan->da_d[ri]->cols; c++)
      gan->da_d[ri]->data[r][c] = -1.0 / (gan->a_d_real[ri]->data[r][c] + 1e-8);

  matrix_t *z = gan->z_d_real[ri];
  for (i = gan->nb_layers - 2; i >= 0; i--)
  {
    if (i != gan->nb_layers - 2)
      mat_dot_(gan->da_d[i], gan->dz_d[i + 1], mat_transpose(gan->w_d[i + 1]));

    switch (gan->act_fn_d[i])
    {
    case LRELU:
      // TODO: not working with z
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dlrelu(gan->z_d_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dsigmoid(mat_sigmoid(z)));
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }
    // pas sûr
    if (i - 1 < 0)
      mat_dot_(gan->dw_d_real[i], mat_transpose(x_real), gan->dz_d[i]);
    else
      mat_dot_(gan->dw_d_real[i], mat_transpose(gan->a_d_fake[i - 1]), gan->dz_d[i]);
    mat_sum_axis0_(gan->db_d_real[i], gan->dz_d[i]);

    z = gan->z_d_fake[i];
  }

  for (r = 0; r < gan->da_d[ri]->rows; r++)
    for (c = 0; c < gan->da_d[ri]->cols; c++)
      gan->da_d[ri]->data[r][c] = 1.0 / (1.0 - gan->a_d_fake[ri]->data[r][c] + 1e-8);

  for (i = gan->nb_layers - 2; i >= 0; i--)
  {
    if (i != gan->nb_layers - 2)
      mat_dot_(gan->da_d[i], gan->dz_d[i + 1], mat_transpose(gan->w_d[i + 1]));

    switch (gan->act_fn_d[i])
    {
    case LRELU:
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dlrelu(gan->z_d_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dsigmoid(mat_sigmoid(gan->z_d_fake[i])));
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }

    if (i - 1 < 0)
      mat_dot_(gan->dw_d_fake[i], mat_transpose(gan->a_g[ri]), gan->dz_d[i]);
    else
      mat_dot_(gan->dw_d_fake[i], mat_transpose(gan->a_d_fake[i - 1]), gan->dz_d[i]);
    mat_sum_axis0_(gan->db_d_fake[i], gan->dz_d[i]);
  }

  matrix_t **dw = (matrix_t **)malloc((gan->nb_layers - 1) * sizeof(*dw));
  matrix_t **db = (matrix_t **)malloc((gan->nb_layers - 1) * sizeof(*db));

  for (i = 0; i < gan->nb_layers - 1; i++)
  {
    dw[i] = mat_zinit(gan->dw_d_fake[i]->rows, gan->dw_d_fake[i]->cols);
    db[i] = mat_zinit(gan->db_d_fake[i]->rows, gan->db_d_fake[i]->cols);

    mat_sum_(dw[i], gan->dw_d_real[i], gan->dw_d_fake[i]);
    mat_sum_(db[i], gan->db_d_real[i], gan->db_d_fake[i]);

    mat_mul_scalar(dw[i], gan->lr);
    mat_sub_(gan->w_d[i], gan->w_d[i], dw[i]);

    mat_mul_scalar(db[i], gan->lr);
    mat_sub_(gan->b_d[i], gan->b_d[i], db[i]);
  }
  /* for (i = 0; i < (gan->nb_layers - 2); i++)
  {
    free(dw[i]);
    free(db[i]);
    dw[i] = NULL;
    db[i] = NULL;
  } */
}

void backward_g_(gan_t *gan, matrix_t *z)
{
  int r, c;

  int i, ri = gan->nb_layers - 2;
  for (r = 0; r < gan->da_d[ri]->rows; r++)
    for (c = 0; c < gan->da_d[ri]->cols; c++)
      gan->da_d[ri]->data[r][c] = -1.0 / (gan->a_d_fake[ri]->data[r][c] + 1e-8);

  for (i = ri; i >= 0; i--)
  {
    if (i != ri)
      mat_dot_(gan->da_d[i], gan->dz_d[i + 1], mat_transpose(gan->w_d[i + 1]));

    switch (gan->act_fn_d[i])
    {
    case LRELU:
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dlrelu(gan->z_d_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(gan->dz_d[i], gan->da_d[i], mat_dsigmoid(mat_sigmoid(gan->z_d_fake[i])));
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }
  }

  matrix_t *dx_d = mat_dot(gan->dz_d[0], mat_transpose(gan->w_d[0]));
  matrix_t *act = dx_d;
  for (i = ri; i >= 0; i--)
  {
    if (i != ri)
      mat_dot_(gan->da_g[i], gan->dz_g[i + 1], mat_transpose(gan->w_g[i + 1]));

    switch (gan->act_fn_g[i])
    {
    case TANH:
      mat_mul_(gan->dz_g[i], act, mat_dtanh(gan->z_g[i]));
      break;
    case LRELU:
      mat_mul_(gan->dz_g[i], act, mat_dlrelu(gan->z_g[i], 0));
      break;
    default:
      printf("Error activation function.\n");
      exit(1);
    }
    act = gan->da_g[MIN(0, i - 1)];

    if (i - 1 < 0)
      mat_dot_(gan->dw_g[i], mat_transpose(z), gan->dz_g[i]);
    else
      mat_dot_(gan->dw_g[i], mat_transpose(gan->a_g[i - 1]), gan->dz_g[i]);

    mat_sum_axis0_(gan->db_g[i], gan->dz_g[i]);
  }

  for (i = 0; i < (gan->nb_layers - 1); i++)
  {
    mat_mul_scalar(gan->dw_g[i], gan->lr);
    mat_sub_(gan->w_g[i], gan->w_g[i], gan->dw_g[i]);

    mat_mul_scalar(gan->db_g[i], gan->lr);
    mat_sub_(gan->b_g[i], gan->b_g[i], gan->db_g[i]);
  }

  // mat_free(act);
  // mat_free(dx_d);
}

void train_gan(cfg_t *cfg, gan_t *gan)
{
  int i, j, bs, il, i_min;
  int ri = gan->nb_layers - 2;
  double losses_d, losses_g, avg_d_fake, avg_d_real;
  losses_d = losses_g = avg_d_fake = avg_d_real = 0.0;

  matrix_t *z = mat_init(cfg->batch_size, gan->input_layer_sz_g);
  matrix_t *x_real = mat_zinit(cfg->batch_size, cfg->x_train->cols);

  for (i = 0; i < gan->epochs; i++)
  {
    for (j = 0; j < cfg->num_batches; j++)
    {
      for (bs = 0; bs < cfg->batch_size; bs++)
        for (il = 0; il < gan->input_layer_sz_g; il++)
          z->data[bs][il] = normal_rand();

      i_min = i * cfg->batch_size;
      mat_copy_(x_real, cfg->x_train, i_min);

      forward_g_(gan, z);
      forward_d_(gan, x_real, 1);
      forward_d_(gan, gan->a_g[ri], 0);

      backward_d_(gan, x_real);
      backward_g_(gan, z);

      losses_d = mat_mean(mat_ce(gan->a_d_fake[ri], gan->a_d_real[ri]));
      losses_g = mat_mean(mat_log(gan->a_d_fake[ri]));
    }
    if (i % 5 == 0)
    {
      printf("epoch:  %d\n", i);
      printf("lr:     %f\n", gan->lr);
      printf("loss_g: %.3f\n", losses_g);
      printf("loss_d: %.3f\n", losses_d);
      printf("avg_df: %.3f\n", mat_mean(gan->a_d_fake[ri]));
      printf("avg_dr: %.3f\n\n", mat_mean(gan->a_d_real[ri]));
      save_mnist_pgm_mat(gan->a_g[ri]);
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