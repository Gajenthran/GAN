/*!
 * \file config.h
 * \brief Fichier header du fichier config.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "mnist.h"
#include "matrix.h"

/* Structure représentant la configuration du programme */
typedef struct config config_t;
struct config
{
  unsigned int batch_sz;
  unsigned int chosen_label;
  unsigned int num_train;
  unsigned int num_batches;
  unsigned int img_sz;
  unsigned int train_sz;
  unsigned int nb_layers;
  unsigned int in_layer_sz_g;
  unsigned int hd_layer_sz_g;
  unsigned int hd_layer_sz_d;
  unsigned int epochs;
  double learning_rate;
  double decay_rate;
  unsigned int *y_train;
  matrix_t *x_train;
};

config_t *init_config(const char *);
void load_mnist_config(config_t *, mnist_t *);
#endif