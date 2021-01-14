/*!
 * \file config.c
 * \brief Fichier s'occupant de la structure de configuration
 * du modèle GAN.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"

// Taille du batch pour l'entraînement
#define BATCH_SZ 64
// Choix du label pour GAN
#define CHOSEN_LABEL 3
// Code ASCII pour l'arobase
#define ASCII_AT 64
// Code ASCII pour l'accolade ouvrante
#define ASCII_BRACE 123

// Hashcode pour les champs de la structure de configuration
// Hashcode pour la taille du batch
#define HASH_BATCH_SZ 210668250183
// Hashcode pour le choix du label
#define HASH_CHOSEN_LABEL 210680089861
// Hashcode pour la taille de l'image
#define HASH_IMG_SZ 6952339998606
// Hashcode pour le nombre de données d'apprentissage
#define HASH_NUM_TRAIN 210690187203
// Hashcode pour le nombre de couches pour le GAN
#define HASH_NB_LAYERS 6952443792245
// Hashcode pour la taille de la couche d'entrée du generator
#define HASH_IN_LAYER_SZ_G 6384152450
// Hashcode pour la taille de la couche cachée du generator
#define HASH_HD_LAYER_SZ_G 6384105623
// Hashcode pour la taille de la couche cachée du discriminator
#define HASH_HD_LAYER_SZ_D 6384105620
// Hashcode pour le coefficient d'apprentissage
#define HASH_LEARNING_RATE 5862499
// Hashcode pour le ratio de décroissance pour le coefficient d'apprentissage
#define HASH_DECAY_RATE 5862235
// Hashcode pour le nombre d'itérations
#define HASH_EPOCHS 6952187271431
// Hashcode pour le verbose
#define HASH_VERBOSE 229443707952891
// Hashcode pour la barre de progression
#define HASH_PBAR 6384389962

/**
 * Fonction de hashing permettant d'obtenir 
 * un hashcode à partir d'une chaîne de caractère.
 * 
 * \param str chaîne de caractère
 * \return hashcode
 */
static unsigned long hash(const char* str)
{
  int c;
  unsigned long hashcode = 5381;

  while ((c = *str++))
    hashcode = ((hashcode << 5) + hashcode) + c;

  return hashcode;
}

/**
 * Initialiser la structure de configuration à partir
 * du fichier passé en paramètre.
 * 
 * \param config_file fichier de configuration
 * \return structure config
 */
config_t* init_config(const char* config_file)
{
  const int MAX = 1024;
  FILE* fp = fopen(config_file, "r");
  if (!fp) {
    fprintf(stderr, "Can't open file %s\n", config_file);
    exit(1);
  }

  char *buf = (char *)malloc(MAX * sizeof(*buf)), *tok, *end;
  assert(buf);

  config_t* cfg = (config_t*)malloc(sizeof *cfg);
  assert(cfg);

  while (!feof(fp)) {
    fgets(buf, MAX, fp);
    if (ferror(fp)) {
      fprintf(stderr, "Error while reading file %s\n", config_file);
      exit(1);
    }

    tok = strtok(buf, "=");
    while (tok != NULL) {
      if (tok != NULL && 
        tok[0] > ASCII_AT && 
        tok[0] < ASCII_BRACE
      ) {
        switch (hash(tok)) {
        case HASH_BATCH_SZ:
          tok = strtok(NULL, "=");
          cfg->batch_sz = atoi(tok);
          break;
        case HASH_NUM_TRAIN:
          tok = strtok(NULL, "=");
          cfg->num_train = atoi(tok);
          break;
        case HASH_CHOSEN_LABEL:
          tok = strtok(NULL, "=");
          cfg->chosen_label = strtod(tok, &end);
          break;
        case HASH_IMG_SZ:
          tok = strtok(NULL, "=");
          cfg->img_sz = atoi(tok);
          break;
        case HASH_NB_LAYERS:
          tok = strtok(NULL, "=");
          cfg->nb_layers = atoi(tok);
          break;
        case HASH_IN_LAYER_SZ_G:
          tok = strtok(NULL, "=");
          cfg->in_layer_sz_g = atoi(tok);
          break;
        case HASH_HD_LAYER_SZ_G:
          tok = strtok(NULL, "=");
          cfg->hd_layer_sz_g = atoi(tok);
          break;
        case HASH_HD_LAYER_SZ_D:
          tok = strtok(NULL, "=");
          cfg->hd_layer_sz_d = atoi(tok);
          break;
        case HASH_LEARNING_RATE:
          tok = strtok(NULL, "=");
          cfg->learning_rate = strtod(tok, &end);
          break;
        case HASH_DECAY_RATE:
          tok = strtok(NULL, "=");
          cfg->decay_rate = strtod(tok, &end);
          break;
        case HASH_EPOCHS:
          tok = strtok(NULL, "=");
          cfg->epochs = atoi(tok);
          break;
        case HASH_VERBOSE:
          tok = strtok(NULL, "=");
          cfg->verbose = atoi(tok);
          break;
        case HASH_PBAR:
          tok = strtok(NULL, "=");
          cfg->progressbar = atoi(tok);
          break;
        default:
          fprintf(stderr, "Error: %s is not a valid parameter.\n", tok);
          exit(0);
        }
      }
      tok = strtok(NULL, "=");
    }
  }
  return cfg;
}

/**
 * Charger les données MNIST pour la structure de configuration,
 * et récupérer les données d'apprentissage.
 * 
 * \param cfg structure config
 * \param mnist structure mnist
 */
void load_mnist_config(config_t* cfg, mnist_t* mnist)
{
  int i, s, j = 0, size = 0;
  for (i = 0; i < cfg->num_train; i++)
    if (mnist->train_label[i] == cfg->chosen_label)
      size++;

  int num_batches = size / cfg->batch_sz;
  size = num_batches * cfg->batch_sz;

  matrix_t* x_train = mat_zinit(size, cfg->img_sz);
  unsigned int* y_train = (unsigned int*)malloc(size * sizeof(*y_train));
  assert(y_train);

  for (i = 0; i < cfg->num_train; i++) {
    // Récupérer seulement le label demandé
    if (mnist->train_label[i] == cfg->chosen_label) {
      for (s = 0; s < cfg->img_sz; s++)
        x_train->data[j * cfg->img_sz + s] = mnist->train_image[i][s];

      y_train[j] = mnist->train_label[i];
      j++;
    }

    if (j == size)
      break;
  }

  // TODO: Shuffle

  cfg->x_train = x_train;
  cfg->y_train = y_train;
  cfg->train_sz = size;
  cfg->num_batches = num_batches;
}