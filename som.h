/*!
 * \file som.h
 * \brief Fichier header de som.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _SOM_H_
#define _SOM_H_

#include "parser.h"
#include "config.h"

/** \brief Structure représentant les neurones */
typedef struct node node_t;
struct node {
  double * w;   // vecteur de données
  char * label; // étiquette
  double act;   // état d'activation
  int * freq;   // fréquence pour chaque labels
};

/** \brief Structure représentant le réseau de neurones */
typedef struct network network_t;
struct network {
  node_t ** map; // réseau, map bidimensionnelle
  double alpha;  // coefficient d'apprentissage
  int nhd_rad;   // rayon de voisinage
};

/** \brief Structure représentant le best match unit */
typedef struct bmu bmu_t;
struct bmu {
  double act; // état d'activation
  int l, c;   // ligne, colonne
};

int *       init_shuffle(int);
void        shuffle(int *, int);
network_t * init_network(data_t *, config_t *);
void        train(network_t *, int *, data_t *, config_t *);
void        label(network_t * net, data_t * data, config_t *);
bmu_t       find_bmu(network_t *, double *, config_t *);
void        apply_nhd(network_t *, double *, bmu_t, config_t *);
double      euclidean_dist(double *, double *, int);
double      my_rand(double min, double max);
void        print_map(network_t *, config_t *);
void        free_shuffle(int *sh);
void        free_network(network_t *net);
#ifdef DEBUG
void        print_net(network_t *, config_t *);
void        print_shuffle(int *, int);
#endif

#endif