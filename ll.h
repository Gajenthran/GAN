/*!
 * \file ll.h
 * \brief Fichier header ll.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef LL_H_
#define LL_H_

#include "som.h"

/** \brief Structure représentant le noeud de la liste chaînée */
typedef struct lnode lnode_t;
struct lnode {
  bmu_t bmu;      // bmu (best match unit)
  lnode_t * next; // élément suivant
};

/** \brief Structure représentant la liste chaînée */
typedef struct list list_t;
struct list {
  lnode_t * head; // tête de la liste
  int size;       // taille de la liste
};

extern list_t * init_list(void);
extern bmu_t    get_bmu_from_list(list_t * l);
extern void     insert_list(list_t * l, bmu_t bmu);
extern void     modify_list(list_t * l, bmu_t bmu);
extern void     print_list(list_t * l);
extern void     free_list(list_t * l);

#endif