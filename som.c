/*!
 * \file som.c
 * \brief Fichier comprenant les fonctionnalités
 * de parsing pour exécuter le modèle de SOM: 
 * la phase d'apprentissage et la phase
 * d'étiquetage.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "som.h"
#include "ll.h"

#define RED   "\x1B[31m"
#define GREEN "\x1B[32m"
#define BLUE  "\x1B[34m"
#define WHITE "\x1B[37m"
#define RESET "\x1B[0m"

/** \brief Initialise le vecteur représentant l'ordre
 * de passage des données lors de la phase d'apprentissage
 *
 * \param size nombre de données
 *
 * \return vecteur représentant l'ordre de passage des données.
 */
int * init_shuffle(int size) {
  srand(time(NULL));
  int i;
  int * sh = (int *)malloc(size * sizeof(*sh));
  assert(sh);

  for(i = 0; i < size; i++)
    sh[i] = i;
  shuffle(sh, size);
  return sh;
}

/** \brief Mélange le vecteur représentant l'ordre de passage des données
 * lors de la phase d'apprentissage
 *
 * \param sh   vecteur représentant l'ordre de passage des données
 * \param size taille du vecteur
 */
void shuffle(int * sh, int size) {
  int i, r;
  for(i = 0; i < size; i++) {
    r = rand() % size;
    if(sh[i] != sh[r]) {
      sh[i] ^= sh[r];
      sh[r] ^= sh[i];
      sh[i] ^= sh[r];
    }
  }
}

/** \brief Initialise le réseau de neurones à partir des
 * des données de la bd, en initialisant le coefficient
 * d'apprentissage, le vecteur de données et le rayon
 * de voisinage.
 *
 * \param data données de la bd
 * \param cfg  données de configuration
 *
 * \return structure de type network_t représentant le
 * réseau de neurones.
 */
network_t * init_network(data_t * data, config_t *cfg) {
  int i, j, l, c;
  double avg, sum;

  network_t * net = (network_t *)malloc(sizeof *net);
  assert(net);
  net->map = (node_t **)malloc(cfg->map_l * sizeof(*net->map));
  assert(net->map);

  for(l = 0; l < cfg->map_l; l++) {
    net->map[l] = (node_t *)malloc(cfg->map_c * sizeof(*net->map[l]));
    assert(net->map[l]);

    for(c = 0; c < cfg->map_c; c++) {
      for(i = 0; i < cfg->nb_val; i++) {
        net->map[l][c].w = (double *)malloc(cfg->nb_val * sizeof(*net->map[l][c].w));
        assert(net->map[l][c].w);
      }

      net->map[l][c].freq = (int *)calloc(0, cfg->nb_label * sizeof(*net->map[l][c].freq));
      assert(net->map[l][c].freq);
    }
  }

  for(i = 0; i < cfg->nb_val; i++) {
    sum = 0;
    for(j = 0; j < cfg->data_sz; j++)
      sum += data[j].v[i];
    avg = sum / cfg->data_sz;

    for(l = 0; l < cfg->map_l; l++)
      for(c = 0; c < cfg->map_c; c++)
        net->map[l][c].w[i] = my_rand(avg - cfg->w_avg_min, avg + cfg->w_avg_max);
  }

  net->alpha = cfg->alpha;
  net->nhd_rad = cfg->nhd_rad;
  return net;
}

/** \brief Phase d'apprentissage des neurones coupé en deux
 * parties: partie compétitive avec la recherche du bmu
 * et partie coopérative avec la diffusion des règles
 * d'apprentissage au voisinage.
 *
 * \param net  réseau de neurones
 * \param sh   vecteur représentant l'ordre de passage des données
 * \param data données
 * \param cfg  données de configuration
 */
void train(network_t * net, int * sh, data_t * data, config_t *cfg) {
  bmu_t bmu;
  int i, l, it, iterations;
  double ph;
  const char * label[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
  for(ph = cfg->ph_1; ph < 1.0; ph += (cfg->ph_2 - cfg->ph_1)) {
    // nombre d'itérations
    iterations = cfg->iter * ph;
    cfg->nhd_rad = ph == cfg->ph_1 ? cfg->nhd_rad : 2;
    cfg->alpha = ph == cfg->ph_1 ? cfg->alpha : cfg->alpha / 10;
    for(it = 0; it < iterations; it++) {
      shuffle(sh, cfg->data_sz);
      // pour tout i appartenant aux données v de la bd
      for(i = 0; i < cfg->data_sz; i++) {
        bmu = find_bmu(net, data[sh[i]].v, cfg);
        apply_nhd(net, data[sh[i]].v, bmu, cfg);

        for(l = 0; l < cfg->nb_label; l++) {
          if(!strcmp(label[l], data[sh[i]].label)) {
            net->map[bmu.l][bmu.c].freq[l]++;
          }
        }
      }

      net->nhd_rad = cfg->nhd_rad * exp(-(double)it / (double)iterations);
      net->alpha = cfg->alpha * (1.0 - ((double)it / (double)iterations));
    }
  }
}

/** \brief Etiqueter les données après la phase d'apprentissage.
 *
 * \param net  réseau de neurones
 * \param data données
 * \param cfg  données de configuration
 */
void label(network_t * net, data_t * data, config_t *cfg) {
  int l, c, i;
  bmu_t bmu;
  double dist;

  for(l = 0; l < cfg->map_l; l++) {
    for(c = 0; c < cfg->map_c; c++) {
      bmu.act = euclidean_dist(net->map[l][c].w, data[0].v, cfg->nb_val);
      bmu.l = 0;
      for(i = 0; i < cfg->data_sz; i++) {
        dist = euclidean_dist(net->map[l][c].w, data[i].v, cfg->nb_val);
        if(bmu.act > dist) {
          bmu.act = dist;
          bmu.l = i;
        }
      }

      // vérifie si le neurone sélectionné n'oscille pas entre deux classes
      int sum = net->map[l][c].freq[0] + net->map[l][c].freq[1] + net->map[l][c].freq[2];
      int find = 0;
      for(i = 0; i < cfg->nb_label; i++) {
        if(net->map[l][c].freq[i] * 100 / (double)sum > cfg->margin_err) {
          net->map[l][c].label = strdup(data[bmu.l].label);
          find = 1;
          break;
        }
      }

      if(!find) { net->map[l][c].label = strdup("not_find"); }
      net->map[l][c].act = bmu.act;
    }
  }
}

/** \brief Applique la règle d'apprentissage selon aux voisins
 * du bmu selon le rayon de voisinage attribué. Concernant la
 * règle d'apprentisage, celle-ci fera abstraction de la fonction
 * de voisinage; ainsi, tous les voisins auront le même taux
 * d'apprentissage que les autres.
 *
 * \param net réseau de neurones
 * \param v   vecteur de données de la bd
 * \param bmu structure représentant le bmu (best match unit)
 * \param cfg données de configuration
 */
void apply_nhd(network_t * net, double * v, bmu_t bmu, config_t *cfg) {
  int i, l, c, l0, c0;
  // pour tout node l, c appartenant à Nhd(i)
  for(l = -net->nhd_rad; l <= net->nhd_rad; l++) {
    for(c = -net->nhd_rad; c <= net->nhd_rad; c++) {
      l0 = bmu.l + l;
      c0 = bmu.c + c;
      if(l0 < 0 || l0 >= cfg->map_l || c0 < 0 || c0 >= cfg->map_c)
        continue;

      for(i = 0; i < cfg->nb_val; i++) {
        net->map[l0][c0].w[i] = net->map[l0][c0].w[i] +
          net->alpha * (v[i] - net->map[l0][c0].w[i]);
      }
    }
  }
}

/** \brief Trouver le bmu (best match unit) en comparant les
 * neurones aux valeurs des données de la bd.
 *
 * \param net réseau de neurones
 * \param v   vecteur de données
 * \param cfg données de configuration
 *
 * \return structure réprésentant le bmu (best match unit)
 */

bmu_t find_bmu(network_t * net, double * v, config_t *cfg) {
  int l, c;
  double dist;
  list_t * bmu_lis = init_list();
  bmu_t bmu = {euclidean_dist(v, net->map[0][0].w, cfg->nb_val), 0, 0};
  insert_list(bmu_lis, bmu);
  for(l = 0; l < cfg->map_l; l++) {
    for(c = 0; c < cfg->map_c; c++) {
      dist = euclidean_dist(v, net->map[l][c].w, cfg->nb_val);

      if(bmu.act > dist) {
        bmu.act = dist;
        bmu.l = l; bmu.c = c;
        modify_list(bmu_lis, bmu);
      }

      if(bmu.act == dist) {
        bmu.l = l; bmu.c = c;
        insert_list(bmu_lis, bmu);
      }
    }
  }

  bmu = get_bmu_from_list(bmu_lis);
  free_list(bmu_lis);
  return bmu;
}

/** \brief Calcule la distance euclidienne de deux vecteurs
 *
 * \param v vecteur v
 * \param w vecteur w
 * \param size taille des vecteurs v et w (partageant la même taille)
 *
 * \return la distance de l'ensemble des deux vecteurs.
 */
double euclidean_dist(double * v, double * w, int size) {
  double sum = 0;
  int i;
  for(i = 0; i < size; i++)
    sum += pow(v[i] - w[i], 2.0);
  return sqrt(sum);
}

/** \brief Fonction retournant un nombre aléatoire entre deux
 * intervalles.
 *
 * \param min interval minimal
 * \param max interval maximal
 *
 * \return Nombre aléatoire entre les intervalles min-max.
 */
double my_rand(double min, double max) {
  return (rand()/(double)RAND_MAX) * (max - min) + min;
}

/** \brief Affiche la map avec les neurones étiquetés.
 *
 * \param net réseau de neurones
 * \param cfg données de configuration
 */
void print_map(network_t * net, config_t * cfg) {
  int l, c;
  printf("Iris-setosa:     "); printf(RED   " o \n"     RESET);
  printf("Iris-versicolor: "); printf(BLUE  " o \n"     RESET);
  printf("Iris-virginica:  "); printf(GREEN " o \n\n"     RESET);
  for(l = 0; l < cfg->map_l; l++) {
    for(c = 0; c < cfg->map_c; c++) {
      if(!strcmp(net->map[l][c].label, "Iris-setosa")) {
        printf(RED " o " RESET);
      } else if(!strcmp(net->map[l][c].label, "Iris-versicolor")) {
        printf(BLUE " o " RESET);
      } else if(!strcmp(net->map[l][c].label, "Iris-virginica")) {
        printf(GREEN " o " RESET);
      } else {
        printf(WHITE " x " RESET);
      }
    }
    printf(RESET "\n"  RESET);
  }
  printf("\n");
}

/** \brief Libère la mémoire pour le shuffle.
 *
 * \param sh vecteur représentant l'ordre de passage des données
 */
void free_shuffle(int *sh) {
  if(sh) {
    free(sh);
    sh = NULL;
  }
}

/** \brief Libère la mémoire pour le réseau de neurones.
 *
 * \param net réseau de neurones
 */
void free_network(network_t *net) {
  if(net) {
    free(net);
    net = NULL;
  }
}

#ifdef DEBUG
void print_net(network_t * net, config_t *cfg) {
  int l, c, i;
  printf("alpha:       %.2f\n", net->alpha);
  printf("n.rad:       %d\n", net->nhd_rad);
  for(l = 0; l < cfg->map_l; l++) {
    for(c = 0; c < cfg->map_c; c++) {
      for(i = 0; i < cfg->nb_val; i++)
        printf("[%d][%d].w[%d]: %.2f\n", l, c, i, net->map[l][c].w[i]);
      printf("\n");
    }
    printf("\n");
  }
}

void print_shuffle(int * sh, int size) {
  int i;
  for(i = 0; i < size; i++)
    printf("%d - ", sh[i]);
  printf("\n");
}
#endif