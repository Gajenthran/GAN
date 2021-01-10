/*!
 * \file som.h
 * \brief Fichier header de som.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _GAN_H_
#define _GAN_H_

#include "config.h"

enum ACT_E
{
  LRELU = 0,
  SIGMOID,
  TANH
};

typedef struct gan_t gan_t;
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

gan_t *init_gan(config_t *);
void forward_g_(gan_t *, matrix_t *);
void forward_d_(gan_t *, matrix_t *, int);
void backward_d_(gan_t *, matrix_t *);
void backward_g_(gan_t *, matrix_t *);
void train_gan(config_t *, gan_t *, mnist_t *);

#endif