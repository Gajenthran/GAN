/*!
 * \file config.h
 * \brief Fichier header du fichier config.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "mnist.h"
#include "matrix.h"

typedef struct config config_t;
/* Structure représentant la configuration pour le GAN */
struct config {
  char verbose; // verbose
  unsigned int progressbar; // barre de progression
  unsigned int batch_sz; // ratio pour le lot
  unsigned int chosen_label; // choix du label
  unsigned int num_train; // nombre de données d'apprentissage
  unsigned int num_batches; // nombre de données pour le lot
  unsigned int img_sz; // taille de l'image
  unsigned int train_sz; // taille du nombre de données
  unsigned int nb_layers; // nombre de couches
  unsigned int in_layer_sz_g; // taille de la couche d'entrée (generator)
  unsigned int hd_layer_sz_g; // taille de la couche cachée (generator)
  unsigned int hd_layer_sz_d; // taille de la couche cachée (discriminator)
  unsigned int epochs; // nombre d'itérations
  double learning_rate; // coefficient d'apprentissage
  double decay_rate; // ratio de décroissance
  unsigned int* y_train; // labels
  matrix_t* x_train; // données d'apprentissage
};

config_t* init_config(const char*);
void load_mnist_config(config_t*, mnist_t*);

#endif