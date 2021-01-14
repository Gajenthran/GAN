/*!
 * \file gan.h
 * \brief Fichier header de gan.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _GAN_H_
#define _GAN_H_

#include "config.h"

/* Enumération pour la fonction d'activation */
enum ACT_E {
  LRELU = 0,
  SIGMOID,
  TANH
};

typedef struct generator generator_t;
/* Structure pour le generator du GAN */
struct generator {
  matrix_t** w; // poids
  matrix_t** b; // biais
  matrix_t** z; // pre-activation
  matrix_t** a; // activation
};

typedef struct discriminator discriminator_t;
/* Structure pour le discriminator du GAN */
struct discriminator {
  matrix_t** w; // poids
  matrix_t** b; // biais
  matrix_t** z_fake; // pre-activation pour le generator
  matrix_t** z_real; // pre-activation pour les données MNIST
  matrix_t** a_fake; // activation pour le generator
  matrix_t** a_real; // pre-activation pour les données MNIST
};

typedef struct der_discriminator der_discriminator_t;
/* Structure pour les dérivées du discriminator du GAN */
struct der_discriminator {
  matrix_t** a; // activation
  matrix_t** z; // pre-activation
  matrix_t* x; //
  matrix_t** w_real; // poids pour le calcul du discriminator avec les données MNIST
  matrix_t** w_fake; // poids pour le calcul du discriminator avec le generator
  matrix_t** b_real; // biais pour le calcul du discriminator avec les données MNIST
  matrix_t** b_fake; // biais pour le calcul du discriminator avec le generator
};

typedef struct gan_t gan_t;
/* Structure pour le modèle GAN */
struct gan_t {
  unsigned int* layers_sz_d; // nombre de neurones dans chaque couche (discriminator)
  unsigned int* layers_sz_g; // nombre de neurones dans chaque couche (generator=
  unsigned int nb_layers; // nombre de couches
  unsigned int epochs; // nombre d'itérations
  unsigned int input_layer_sz_g; // taille de la couche d'entrée (generator)
  unsigned int hidden_layer_sz_g; // taille de la couche cachée (generator)
  unsigned int hidden_layer_sz_d; // taille de la couche cachée (discriminator)
  int* act_fn_d; // id pour les fonctions d'activation pour chaque couche (discriminator)
  int* act_fn_g; // id pour les fonctions d'activation pour chaque couche (generator)
  double lr; // coefficient d'apprentissage
  double dr; // ratio de décroissance

  generator_t* g; // generator
  discriminator_t* d; // discriminator
  generator_t* der_g; // dérivées pour le generator
  der_discriminator_t* der_d; // dérivées pour le discriminator
};

gan_t* init_gan(config_t*);
void forward_generator(gan_t*, matrix_t*);
void forward_discriminator(gan_t*, matrix_t*, int);
void backward_discriminator(gan_t*, matrix_t*);
void backward_generator(gan_t*, matrix_t*);
void train_gan(config_t*, gan_t*, mnist_t*);

#endif