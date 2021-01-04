/*!
 * \file ll.c
 * \brief Fichier comprenant les fonctionnalités
 * de la liste chaînée
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "ll.h"

/** \brief Initialise la liste chaînée.
 *
 * \return liste chaînée de type list_t
 */
list_t * init_list(void) {
  list_t * l = (list_t *)malloc(sizeof *l);
  assert(l);
  l->head = NULL;
  l->size = 0;
  return l;
}

/** \brief Retourne de manière aléatoire un bmu de
 * la liste.
 *
 * \param l liste chaînée
 * \return bmu (best match unit)
 */
bmu_t get_bmu_from_list(list_t * l) {
  srand(time(NULL));
  int r = (rand() % l->size), it = 0;
  lnode_t * n;
  while(l->head) {
    n = l->head;
    l->head = l->head->next;
    it++;
    if(it >= r) break;
  }

  bmu_t bmu = n->bmu;

  // dernière utilisation de la liste chaînée
  free_list(l);
  return bmu;
}

/** \brief Insérer un élément dans la liste
 *
 * \param l   liste chaînée
 * \param bmu bmu (best match unit)
 */
void insert_list(list_t * l, bmu_t bmu) {
  lnode_t * n = (lnode_t *)malloc(sizeof *n);
  assert(n);
  n->bmu = bmu;
  n->next = l->head;
  l->head = n;
  l->size++;
}

/** \brief Modifier la liste en supprimer tous les
 * éléments de la liste chaînée, excepté le premier
 * élément qui sera remplacé par le nouveau bmu donné
 * en paramètre.
 *
 * \param l   liste chaînée
 * \param bmu bmu (best match unit)
 */
void modify_list(list_t * l, bmu_t bmu) {
  if(!l)
    return;

  if(l->head)
    l->head->bmu = bmu;

  while(l->head) {
    lnode_t * n = l->head;
    l->head = l->head->next;
    free(n);
  }
  l->size = 1;
}

/** \brief Libère la liste chaînée
 *
 * \param l liste chaînée
 */
void free_list(list_t * l) {
  if(!l)
    return;

  while(l->head) {
    lnode_t * n = l->head;
    l->head = l->head->next;
    free(n);
  }
  l->size = 0;
  l = NULL;
}

#ifdef DEBUG
void print_list(list_t * l) {
  if(!l || !l->head)
    return;

  lnode_t * h = l->head;
  while(h) {
    printf("%d, %d -> ", h->bmu.l, h->bmu.c);
    h = h->next;
  }
  printf("NULL\n");
}
#endif