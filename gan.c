
/*!
 * \file gan.c
 * \brief Fichier comprenant le modèle GAN avec 
 * le discriminator et le generator.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "gan.h"

// Minimum entre deux nombres
#define MIN(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

// Constante 2 * PI
#define _2PI 6.28

// Constante pour fixer l'affichage a chaque 'n' iteration
#define PRINT_EP 5

/**
 * Générer un nombre aléatoire avec une distribution normale.
 * \return nombre aléatoire
 */
static double normal_rand(void)
{
  double v1 = ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
  double v2 = ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
  return cos(_2PI * v2) * sqrt(-2. * log(v1));
}

/**
 * Initialiser le generator pour le GAN.
 * 
 * \param cfg structure config
 * \param layers_sz_g taille de la couche d'entrée (generator)
 * \return la structure generator
 */
static generator_t* init_generator(config_t* cfg, unsigned int* layers_sz_g)
{
  matrix_t** w_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*w_g));
  assert(w_g);
  matrix_t** b_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*b_g));
  assert(b_g);
  matrix_t** z_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*z_g));
  assert(z_g);
  matrix_t** a_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*a_g));
  assert(a_g);

  int r, c, i, g_rows;
  for (i = 0; i < cfg->nb_layers - 1; i++) {
    g_rows = (i == 0) ? cfg->batch_sz : a_g[i - 1]->rows;

    w_g[i] = mat_zinit(layers_sz_g[i], layers_sz_g[i + 1]);
    b_g[i] = mat_zinit(1, layers_sz_g[i + 1]);
    a_g[i] = mat_zinit(g_rows, w_g[i]->cols);
    z_g[i] = mat_zinit(g_rows, w_g[i]->cols);

    for (r = 0; r < layers_sz_g[i]; r++)
      for (c = 0; c < layers_sz_g[i + 1]; c++)
        w_g[i]->data[r * w_g[i]->cols + c] = normal_rand() * sqrt(2.0 / layers_sz_g[i]);
  }

  generator_t* gen = (generator_t*)malloc(sizeof(*gen));
  assert(gen);

  gen->w = w_g;
  gen->b = b_g;
  gen->z = z_g;
  gen->a = a_g;

  return gen;
}

/**
 * Initialiser les dérivées pour le generator du GAN, pour stocker
 * les matrices et conserver de la mémoire.
 * 
 * \param cfg structure config
 * \param layers_sz_g taille de la couche d'entrée (generator)
 * \param gen structure pour le generator
 * \return la structure generator pour les derivées
 */
static generator_t* init_der_generator(config_t* cfg, unsigned int* layers_sz_g, generator_t* gen)
{
  matrix_t** da_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*da_g));
  assert(da_g);
  matrix_t** dz_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*dz_g));
  assert(dz_g);
  matrix_t** dw_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*dw_g));
  assert(dw_g);
  matrix_t** db_g = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*db_g));
  assert(db_g);

  int i, g_rows;
  for (i = 0; i < cfg->nb_layers - 1; i++) {
    g_rows = (i == 0) ? cfg->batch_sz : gen->a[i - 1]->rows;

    da_g[i] = mat_zinit(g_rows, gen->w[i]->cols);
    dz_g[i] = mat_zinit(g_rows, gen->w[i]->cols);
    dw_g[i] = mat_zinit(layers_sz_g[i], layers_sz_g[i + 1]);
    db_g[i] = mat_zinit(1, layers_sz_g[i + 1]);
  }

  generator_t* der_g = (generator_t*)malloc(sizeof(*gen));
  assert(gen);

  der_g->w = dw_g;
  der_g->b = db_g;
  der_g->z = dz_g;
  der_g->a = da_g;

  return der_g;
}

/**
 * Initialiser le discriminator pour le GAN.
 * 
 * \param cfg structure config
 * \param layers_sz_d taille de la couche d'entrée (discriminator)
 * \return la structure discriminator
 */
static discriminator_t* init_discriminator(config_t* cfg, unsigned int* layers_sz_d)
{
  matrix_t** w_d = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*w_d));
  assert(w_d);
  matrix_t** b_d = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*b_d));
  assert(b_d);
  matrix_t** z_d_fake = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*z_d_fake));
  assert(z_d_fake);
  matrix_t** z_d_real = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*z_d_real));
  assert(z_d_real);
  matrix_t** a_d_fake = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*a_d_fake));
  assert(a_d_fake);
  matrix_t** a_d_real = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*a_d_real));
  assert(a_d_real);

  int r, c, i, d_rows;
  for (i = 0; i < cfg->nb_layers - 1; i++) {
    d_rows = (i == 0) ? cfg->batch_sz : a_d_real[i - 1]->rows;

    w_d[i] = mat_zinit(layers_sz_d[i], layers_sz_d[i + 1]);
    b_d[i] = mat_zinit(1, layers_sz_d[i + 1]);

    a_d_fake[i] = mat_zinit(d_rows, w_d[i]->cols);
    a_d_real[i] = mat_zinit(d_rows, w_d[i]->cols);

    z_d_fake[i] = mat_zinit(d_rows, w_d[i]->cols);
    z_d_real[i] = mat_zinit(d_rows, w_d[i]->cols);

    for (r = 0; r < layers_sz_d[i]; r++)
      for (c = 0; c < layers_sz_d[i + 1]; c++)
        w_d[i]->data[r * w_d[i]->cols + c] = normal_rand() * sqrt(2.0 / layers_sz_d[i]);
  }

  discriminator_t* dis = (discriminator_t*)malloc(sizeof(*dis));
  assert(dis);

  dis->w = w_d;
  dis->b = b_d;
  dis->z_fake = z_d_fake;
  dis->a_fake = a_d_fake;
  dis->z_real = z_d_real;
  dis->a_real = a_d_real;

  return dis;
}

/**
 * Initialiser les dérivées pour le discriminator du GAN, pour stocker
 * les matrices et conserver de la mémoire.
 * 
 * \param cfg structure config
 * \param layers_sz_d taille de la couche d'entrée (discriminator)
 * \param dis structure pour le discriminator
 * \return la structure der_discriminator pour les derivées
 */
static der_discriminator_t* init_der_discriminator(config_t* cfg, unsigned int* layers_sz_d, discriminator_t* dis)
{
  matrix_t** da_d = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*da_d));
  assert(da_d);
  matrix_t** dz_d = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*dz_d));
  assert(dz_d);
  matrix_t** dw_d_real = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*dw_d_real));
  assert(dw_d_real);
  matrix_t** dw_d_fake = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*dw_d_fake));
  assert(dw_d_fake);
  matrix_t** db_d_real = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*db_d_real));
  assert(db_d_real);
  matrix_t** db_d_fake = (matrix_t**)malloc((cfg->nb_layers - 1) * sizeof(*db_d_fake));
  assert(db_d_fake);

  int i, d_rows;
  for (i = 0; i < cfg->nb_layers - 1; i++) {
    d_rows = (i == 0) ? cfg->batch_sz : dis->a_real[i - 1]->rows;

    da_d[i] = mat_zinit(d_rows, dis->w[i]->cols);
    dz_d[i] = mat_zinit(d_rows, dis->w[i]->cols);
    dw_d_real[i] = mat_zinit(layers_sz_d[i], layers_sz_d[i + 1]);
    dw_d_fake[i] = mat_zinit(layers_sz_d[i], layers_sz_d[i + 1]);
    db_d_real[i] = mat_zinit(1, layers_sz_d[i + 1]);
    db_d_fake[i] = mat_zinit(1, layers_sz_d[i + 1]);
  }

  der_discriminator_t* der_d = (der_discriminator_t*)malloc(sizeof(*der_d));
  assert(der_d);

  der_d->a = da_d;
  der_d->z = dz_d;
  der_d->w_real = dw_d_real;
  der_d->w_fake = dw_d_fake;
  der_d->b_real = db_d_real;
  der_d->b_fake = db_d_fake;
  der_d->x = mat_zinit(dz_d[0]->rows, dis->w[0]->rows);

  return der_d;
}

/**
 * Initialiser le modèle GAN avec les paramètres de config.
 * \return structure GAN
 */
gan_t* init_gan(config_t* cfg)
{
  unsigned int* layers_sz_d = (unsigned int*)malloc(cfg->nb_layers * sizeof(*layers_sz_d));
  assert(layers_sz_d);
  unsigned int* layers_sz_g = (unsigned int*)malloc(cfg->nb_layers * sizeof(*layers_sz_g));
  assert(layers_sz_g);

  int* act_fn_g = (int*)malloc((cfg->nb_layers - 1) * sizeof(*act_fn_g));
  assert(act_fn_g);
  int* act_fn_d = (int*)malloc((cfg->nb_layers - 1) * sizeof(*act_fn_d));
  assert(act_fn_d);

  // TODO: Automatiser avec une boucle
  act_fn_g[0] = LRELU;
  act_fn_g[1] = TANH;
  act_fn_d[0] = LRELU;
  act_fn_d[1] = SIGMOID;

  layers_sz_d[0] = MNIST_SIZE;
  layers_sz_d[1] = cfg->hd_layer_sz_d;
  layers_sz_d[2] = 1;

  layers_sz_g[0] = cfg->in_layer_sz_g;
  layers_sz_g[1] = cfg->hd_layer_sz_g;
  layers_sz_g[2] = MNIST_SIZE;

  // generator
  generator_t* gen = init_generator(cfg, layers_sz_g);
  // discriminator
  discriminator_t* dis = init_discriminator(cfg, layers_sz_d);
  // derivées pour le generator
  generator_t* der_g = init_der_generator(cfg, layers_sz_g, gen);
  // derivées pour le discriminator
  der_discriminator_t* der_d = init_der_discriminator(cfg, layers_sz_d, dis);

  gan_t* gan = (gan_t*)malloc(1 * sizeof(*gan));
  assert(gan);

  gan->layers_sz_d = layers_sz_d;
  gan->layers_sz_g = layers_sz_g;
  gan->act_fn_g = act_fn_g;
  gan->act_fn_d = act_fn_d;
  gan->nb_layers = cfg->nb_layers;
  gan->hidden_layer_sz_d = cfg->hd_layer_sz_d;
  gan->hidden_layer_sz_g = cfg->hd_layer_sz_g;
  gan->input_layer_sz_g = cfg->in_layer_sz_g;
  gan->lr = cfg->learning_rate;
  gan->dr = cfg->decay_rate;
  gan->epochs = cfg->epochs;

  gan->g = gen;
  gan->d = dis;
  gan->der_g = der_g;
  gan->der_d = der_d;

  return gan;
}

/**
 * Génère une image par rapport au label demandé
 * avec le generator, à partir de données bruitées.
 * \param gan structure GAN
 * \param z données bruitées
 */
void forward_generator(gan_t* gan, matrix_t* z)
{
  int i;
  matrix_t* act = z;
  generator_t* gen = gan->g;
  for (i = 0; i < gan->nb_layers - 1; i++) {
    mat_sum_z_act(gen->z[i], act, gen->w[i], gen->b[i]);

    switch (gan->act_fn_g[i]) {
    case LRELU:
      mat_lrelu_(gen->a[i], gen->z[i], 0);
      break;
    case TANH:
      mat_tanh_(gen->a[i], gen->z[i]);
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }
    act = gen->a[i];
  }
}

/**
 * Prédire si le label des images avec le discriminator,
 * en cherchant à vérifier la ressemblance des images généres
 * avec le générator par rapport aux images sources.
 * \param gan structure GAN
 * \param x image générée par le generator / image source
 * \param real booléen pour l'image générée ou source
 */
void forward_discriminator(gan_t* gan, matrix_t* x, int real)
{
  discriminator_t* dis = gan->d;
  matrix_t** z = real ? dis->z_real : dis->z_fake;
  matrix_t** a = real ? dis->a_real : dis->a_fake;
  matrix_t* act = x;

  int i;
  for (i = 0; i < gan->nb_layers - 1; i++) {
    mat_sum_z_act(z[i], act, dis->w[i], dis->b[i]);

    switch (gan->act_fn_d[i]) {
    case LRELU:
      mat_lrelu_(a[i], z[i], 1e-2);
      break;
    case SIGMOID:
      mat_sigmoid_(a[i], z[i]);
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }
    act = a[i];
  }
}

/**
 * Propagation en arrière du discriminator pour qu'il apprenne
 * les caractéristiques des données et améliorer ses performances.
 * 
 * \param gan la structure gan
 * \param x_real données d'apprentissage 
 */
void backward_discriminator(gan_t* gan, matrix_t* x_real)
{
  int i, r, c, out = gan->nb_layers - 2;

  generator_t* gen = gan->g;
  discriminator_t* dis = gan->d;
  der_discriminator_t* der_d = gan->der_d;

  // Gradient pour la donnée d'entrée réelle (MNIST)
  for (r = 0; r < der_d->a[out]->rows; r++)
    for (c = 0; c < der_d->a[out]->cols; c++)
      der_d->a[out]->data[r * der_d->a[out]->cols + c] = -1.0 / (dis->a_real[out]->data[r * dis->a_real[out]->cols + c] + 1e-8);

  matrix_t* z = dis->z_real[out];
  matrix_t* act = NULL;

  for (i = gan->nb_layers - 2; i >= 0; i--) {
    if (i != gan->nb_layers - 2) {
      mat_dot_(der_d->a[i], der_d->z[i + 1], dis->w[i + 1], RIGHT_TRANSPOSE);
      z = dis->z_fake[i];
    }

    switch (gan->act_fn_d[i]) {
    case LRELU:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dlrelu(dis->z_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dsigmoid(mat_sigmoid(z)));
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }

    act = i - 1 < 0 ? x_real : dis->a_fake[i - 1];
    mat_dot_(der_d->w_real[i], act, der_d->z[i], LEFT_TRANSPOSE);
    mat_sum_axis0_(der_d->b_real[i], der_d->z[i]);
  }

  // Gradient pour la donnée d'entrée fausse (généré par le GAN)
  for (r = 0; r < der_d->a[out]->rows; r++)
    for (c = 0; c < der_d->a[out]->cols; c++)
      der_d->a[out]->data[r * der_d->a[out]->cols + c] = 1.0 / (1.0 - dis->a_fake[out]->data[r * dis->a_fake[out]->cols + c] + 1e-8);

  act = NULL;

  for (i = gan->nb_layers - 2; i >= 0; i--) {
    if (i != gan->nb_layers - 2)
      mat_dot_(der_d->a[i], der_d->z[i + 1], dis->w[i + 1], RIGHT_TRANSPOSE);

    switch (gan->act_fn_d[i]) {
    case LRELU:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dlrelu(dis->z_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dsigmoid(mat_sigmoid(dis->z_fake[i])));
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }

    act = i - 1 < 0 ? gen->a[out] : dis->a_fake[i - 1];
    mat_dot_(der_d->w_fake[i], act, der_d->z[i], LEFT_TRANSPOSE);
    mat_sum_axis0_(der_d->b_fake[i], der_d->z[i]);
  }

  matrix_t** dw = (matrix_t**)malloc((gan->nb_layers - 1) * sizeof(*dw));
  assert(dw);
  matrix_t** db = (matrix_t**)malloc((gan->nb_layers - 1) * sizeof(*db));
  assert(db);

  for (i = 0; i < gan->nb_layers - 1; i++) {
    dw[i] = mat_zinit(der_d->w_fake[i]->rows, der_d->w_fake[i]->cols);
    db[i] = mat_zinit(der_d->b_fake[i]->rows, der_d->b_fake[i]->cols);

    // Combinaison des deux images (réelle et fausse)
    mat_sum_(dw[i], der_d->w_real[i], der_d->w_fake[i]);
    mat_sum_(db[i], der_d->b_real[i], der_d->b_fake[i]);

    // SGD pour mettre à jour les poids et les biais
    mat_mul_scalar(dw[i], gan->lr);
    mat_sub_(dis->w[i], dis->w[i], dw[i]);

    mat_mul_scalar(db[i], gan->lr);
    mat_sub_(dis->b[i], dis->b[i], db[i]);
  }

  for (i = 0; i < gan->nb_layers - 2; i++) {
    mat_free(dw[i]);
    mat_free(db[i]);
    dw[i] = NULL;
    db[i] = NULL;
  }
}

/**
 * Propagation en arrière du generator pour qu'il apprenne
 * les caractéristiques des données et améliorer ses performances
 * Son but étant de se calquer aux données MNIST pour tromper le 
 * discriminator.
 * 
 * \param gan la structure gan
 * \param z donnée bruitée
 */
void backward_generator(gan_t* gan, matrix_t* z)
{
  int i, r, c, out = gan->nb_layers - 2;

  generator_t* gen = gan->g;
  discriminator_t* dis = gan->d;
  der_discriminator_t* der_d = gan->der_d;
  generator_t* der_g = gan->der_g;

  // Propagation en arrière du discriminator
  // Gradient pour la donnée d'entrée fausse (généré par le GAN)
  for (r = 0; r < der_d->a[out]->rows; r++)
    for (c = 0; c < der_d->a[out]->cols; c++)
      der_d->a[out]->data[r * der_d->a[out]->cols + c] = -1.0 / (dis->a_fake[out]->data[r * dis->a_fake[out]->cols + c] + 1e-8);

  for (i = out; i >= 0; i--) {
    if (i != out)
      mat_dot_(der_d->a[i], der_d->z[i + 1], dis->w[i + 1], RIGHT_TRANSPOSE);

    switch (gan->act_fn_d[i]) {
    case LRELU:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dlrelu(dis->z_fake[i], 1e-2));
      break;
    case SIGMOID:
      mat_mul_(der_d->z[i], der_d->a[i], mat_dsigmoid(mat_sigmoid(dis->z_fake[i])));
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }
  }

  // Propagation en arrière du generator
  // Gradient pour la donnée d'entrée fausse (généré par le GAN)
  mat_dot_(der_d->x, der_d->z[0], dis->w[0], RIGHT_TRANSPOSE);
  matrix_t* act_der_g = der_d->x;
  matrix_t* act_gen = NULL;

  for (i = out; i >= 0; i--) {
    if (i != out)
      mat_dot_(der_g->a[i], der_g->z[i + 1], gen->w[i + 1], RIGHT_TRANSPOSE);

    switch (gan->act_fn_g[i]) {
    case TANH:
      mat_mul_(der_g->z[i], act_der_g, mat_dtanh(gen->z[i]));
      break;
    case LRELU:
      mat_mul_(der_g->z[i], act_der_g, mat_dlrelu(gen->z[i], 0));
      break;
    default:
      fprintf(stderr, "Error: invalid activation function. \n");
      exit(1);
    }

    act_gen = (i - 1 < 0) ? z : gen->a[i - 1];
    mat_dot_(der_g->w[i], act_gen, der_g->z[i], LEFT_TRANSPOSE);
    mat_sum_axis0_(der_g->b[i], der_g->z[i]);
    act_der_g = der_g->a[MIN(0, i - 1)];
  }

  // SGD pour mettre à jour les poids et les biais
  for (i = 0; i < gan->nb_layers - 1; i++) {
    mat_mul_scalar(der_g->w[i], gan->lr);
    mat_sub_(gen->w[i], gen->w[i], der_g->w[i]);

    mat_mul_scalar(der_g->b[i], gan->lr);
    mat_sub_(gen->b[i], gen->b[i], der_g->b[i]);
  }
}

/**
 * Afficher la barre de progression.
 * 
 * \param cfg structure config
 * \param epoch itération actuelle
 */
static inline void print_progressbar(int current_epoch, int print_epoch, int epochs)
{
  int b;

  int nb_bar = current_epoch % print_epoch;
  int filled_bar = nb_bar == 0 ? print_epoch : nb_bar;
  int rem_bar = print_epoch - filled_bar;

  printf("|");
  for (b = 0; b < filled_bar; b++)
    printf(" # ");
  for (b = 0; b < rem_bar; b++)
    printf(" _ ");
  printf("|  [%d/%d] -- (%d/%d) \n\n", filled_bar, print_epoch, current_epoch, epochs);
}

/**
 * Afficher les paramètres pour chaque 'i' iteration: les erreurs, le coefficient,
 * résultat affiché par notre modèle generator.
 * 
 * \param mnist structure mnist
 * \param gan structure gan
 * \param epoch itération actuel de la phase d'apprentissage
 * \param loss_d perte pour le discriminant
 * \param loss_g perte pour le generator
 */
static inline void print_loss(mnist_t* mnist, gan_t* gan, int epoch, matrix_t* loss_d, matrix_t* loss_g)
{
  int out = gan->nb_layers - 2;
  generator_t* gen = gan->g;
  discriminator_t* dis = gan->d;

  mat_ce_(loss_d, dis->a_fake[out], dis->a_real[out]);
  mat_log_(loss_g, dis->a_fake[out]);

  printf("- Epoch n.%d \n", epoch);
  printf(" * lr:     %f\n", gan->lr);
  printf(" * loss_g: %.3f\n", mat_mean(loss_g));
  printf(" * loss_d: %.3f\n", mat_mean(loss_d));
  save_mnist_pgm_mat(gen->a[out], mnist);
  printf("\n");
}

/**
 * Générer du bruit pour l'apprentissage du generator.
 * 
 * \param z matrice pour le stockage du bruit
 */
static inline void generate_noise(matrix_t* z)
{
  int n;
  for (n = 0; n < z->rows * z->cols; n++)
    z->data[n] = normal_rand();
}

/**
 * Entraîner le modèle GAN, avec la propagation en avant
 * du generator et celle du discriminator (avec les données
 * réelles "MNIST" et fausses "GAN"), suivi d'une propagation en arrière
 * du generator et du discriminator.
 * 
 * \param cfg structure config
 * \param gan structure gan
 * \param mnist structure mnist
 */
void train_gan(config_t* cfg, gan_t* gan, mnist_t* mnist)
{
  int i, j;
  int out = gan->nb_layers - 2;

  generator_t* gen = gan->g;
  discriminator_t* dis = gan->d;

  matrix_t* z = mat_zinit(cfg->batch_sz, gan->input_layer_sz_g);
  matrix_t* x_real = mat_zinit(cfg->batch_sz, cfg->x_train->cols);
  matrix_t* loss_d = mat_zinit(dis->a_fake[out]->rows, dis->a_real[out]->cols);
  matrix_t* loss_g = mat_zinit(dis->a_fake[out]->rows, dis->a_fake[out]->cols);

  for (i = 0; i < gan->epochs; i++) {
    for (j = 0; j < cfg->num_batches; j++) {
      generate_noise(z);
      mat_copy_(x_real, cfg->x_train, i * cfg->batch_sz);

      forward_generator(gan, z);
      forward_discriminator(gan, x_real, 1);
      forward_discriminator(gan, gen->a[out], 0);

      backward_discriminator(gan, x_real);
      backward_generator(gan, z);
    }

    if (cfg->progressbar)
      print_progressbar(i, PRINT_EP, gan->epochs);

    if (cfg->verbose && i % PRINT_EP == 0)
      print_loss(mnist, gan, i, loss_d, loss_g);

    gan->lr = gan->lr * (1.0 / (1.0 + gan->dr * i));
  }

  mat_free(z);
  mat_free(x_real);
  mat_free(loss_d);
  mat_free(loss_g);
}