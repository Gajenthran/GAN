/*!
 * \file config.h
 * \brief Fichier header du fichier config.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#define CONFIG_FILE "som.cfg"
#define ASCII_AT 64
#define ASCII_BRACE 123

/* Structure représentant la configuration du programme */
typedef struct config config_t;
struct config {
  int nhd_rad;      // rayon de voisinage
  int iter;         // nombre d'itérations
  int map_l, map_c; // dimension de la map
  int nb_val;       // nombre de valeurs dans les données
  int data_sz;      // nombre de données
  int nb_label;     // nombre de labels
  double alpha;     // coefficient d'apprentissage
  double w_avg_min; // inter min pour w
  double w_avg_max; // interval max pour w
  float margin_err; // marge d'erreur
  float ph_1, ph_2; // phase 1 et phase 2 (rapport de la phase pour les itérations)
};

#endif