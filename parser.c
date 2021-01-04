/*!
 * \file parser.c
 * \brief Fichier comprenant les fonctionnalités
 * de parsing pour parser les données en entrée et
 * le fichier config
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parser.h"


/** \brief Lire le fichiers de données, tokenizer son
 * contenu où chaque valeur est séparée par une virgule
 * placer les éléments dans la struct data_t
 *
 * \param filename nom du fichier
 * \param cfg      données de configuration
 *
 * \return la structure de forme data_t qui représente
 * les données formalisées
 */
data_t * read_file(char * filename, config_t * cfg) {
  const int MAX = 1024;
  FILE * fp = fopen(filename, "r");
  if(!fp) {
    fprintf(stderr, "Can't open file %s\n", filename);
    exit(1);
  }

  int line = 0, j = 0;
  char * buf = (char *)malloc(MAX * sizeof(*buf)), * tok, * end;
  assert(buf);
  data_t * data = (data_t *)malloc(MAX * sizeof(*data));
  assert(data);

  while(!feof(fp)) {
    fgets(buf, MAX, fp);
    if(ferror(fp)) {
      fprintf( stderr, "Error while reading file %s\n", filename);
      exit(1);
    }

    // tokenizer la ligne récupérée par fgets
    char * label;
    tok = strtok(buf, ",");
    data[line].v = (double *)malloc(cfg->nb_val * sizeof(*data[line].v));
    assert(data[line].v);

    j = 0;
    while(tok != NULL) {
      if(j < cfg->nb_val)
        data[line].v[j++] = strtod(tok, &end);
      label = tok;
      tok = strtok(NULL, ",");
    }
    label = strtok(label, "\n");
    data[line++].label = strdup(label);
  }

  cfg->data_sz = line;
  return data;
}

/** \brief Normalise les données
 *
 * \param data ensemble de données
 * \param cfg  données de configuration
 */
void normalize(data_t * data, config_t * cfg) {
  int i, j;
  double sum;

  for(i = 0; i < cfg->data_sz; i++) {
    sum = 0;
    for(j = 0; j < cfg->nb_val; j++)
      sum += pow(data[i].v[j], 2.0);
    data[i].norm = sqrt(sum);
    for(j = 0; j < cfg->nb_val; j++)
      data[i].v[j] /= data[i].norm;
  }
}

/** \brief Initialise les données de
 * configuration
 *
 * \param filename fichier de configuration
 * \return return la structure de configuration
 * config_t
 */
config_t * init_config(char * filename) {
  const int MAX = 1024;
  FILE * fp = fopen(filename, "r");
  if(!fp) {
    fprintf(stderr, "Can't open file %s\n", filename);
    exit(1);
  }

  char * buf = (char *)malloc(MAX * sizeof(*buf)), * tok, * end;
  assert(buf);
  config_t * cfg = (config_t *)malloc(sizeof *cfg);
  assert(cfg);

  while(!feof(fp)) {
    fgets(buf, MAX, fp);
    if(ferror(fp)) {
      fprintf(stderr, "Error while reading file %s\n", filename);
      exit(1);
    }

    tok = strtok(buf, "=");
    while(tok != NULL) {
      if(
        tok != NULL && 
        tok[0] > ASCII_AT && 
        tok[0] < ASCII_BRACE
      ) {
        if(!strcmp(tok, "ALPHA")) {
          tok = strtok(NULL, "=");
          cfg->alpha = strtod(tok, &end);
        } else if(!strcmp(tok, "WAVG_MIN")) {
          tok = strtok(NULL, "=");
          cfg->w_avg_min = strtod(tok, &end);
        } else if(!strcmp(tok, "WAVG_MAX")) {
          tok = strtok(NULL, "=");
          cfg->w_avg_max = strtod(tok, &end);
        } else if(!strcmp(tok, "NHD_RAD")) {
          tok = strtok(NULL, "=");
          cfg->nhd_rad = atoi(tok);
        } else if(!strcmp(tok, "ITER")) {
          tok = strtok(NULL, "=");
          cfg->iter = atoi(tok);
        } else if(!strcmp(tok, "MAP_L")) {
          tok = strtok(NULL, "=");
          cfg->map_l = atoi(tok);
        } else if(!strcmp(tok, "MAP_C")) {
          tok = strtok(NULL, "=");
          cfg->map_c = atoi(tok);
        } else if(!strcmp(tok, "NB_VAL")) {
          tok = strtok(NULL, "=");
          cfg->nb_val = atoi(tok);
        } else if(!strcmp(tok, "MARG_ERR")) {
          tok = strtok(NULL, "=");
          cfg->margin_err = strtod(tok, &end);
        } else if(!strcmp(tok, "PH_1")) {
          tok = strtok(NULL, "=");
          cfg->ph_1 = strtod(tok, &end);
        } else if(!strcmp(tok, "PH_2")) {
          tok = strtok(NULL, "=");
          cfg->ph_2 = strtod(tok, &end);
        } else if(!strcmp(tok, "NB_LABEL")) {
          tok = strtok(NULL, "=");
          cfg->nb_label = atoi(tok);
        } else {
          fprintf( stderr, "Error while reading file %s\n", filename);
          exit(1);
        }
      }
      tok = strtok(NULL, "=");
    }
  }
  return cfg;
}

/** \brief Libère la mémoire pour la configuration.
 *
 * \param cfg données de configuration
 */
void free_config(config_t * cfg) {
  if(cfg) {
    free(cfg);
    cfg = NULL;
  }
}

/** \brief Libère les données de la bd.
 *
 * \param filename données de la bd
 * config_t
 */
void free_data(data_t * data) {
  if(data) {
    free(data);
    data = NULL;
  }
}

#ifdef DEBUG
void print_data(data_t * data, config_t * cfg) {
  int i, j;
  for(i = 0; i < cfg->data_sz; i++) {
    for(j = 0; j < cfg->nb_val; j++) {
      printf("%.1f,", data[i].v[j]);
    }
    printf("%s\n", data[i].label);
  }
}

void print_config(config_t * cfg) {
  printf("nb_val:  %d\n", cfg->nb_val);
  printf("data_sz: %d\n", cfg->data_sz);
  printf("n_label: %d\n", cfg->nb_label);
}
#endif