/*!
 * \file mnist.c
 * \brief Fichier s'occupant de la gestion des images MNIST
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include "config.h"

/**
 * Lire le fichier MNIST et récupérer les données (de type char).
 * \param file fichier MNIST 
 * \param num_data nombre de données
 * \param len_info taille d'une donnée
 * \param n nombre de valeur d'une image
 * \param data données
 * \param buf buffer
 */
void fliplong(unsigned char *ptr)
{
  register unsigned char val;

  // Swap 1st and 4th bytes
  val = *(ptr);
  *(ptr) = *(ptr + 3);
  *(ptr + 3) = val;

  // Swap 2nd and 3rd bytes
  ptr += 1;
  val = *(ptr);
  *(ptr) = *(ptr + 1);
  *(ptr + 1) = val;
}

/**
 * Lire le fichier MNIST et récupérer les données (de type char).
 * \param file fichier MNIST 
 * \param num_data nombre de données
 * \param len_info taille d'une donnée
 * \param n nombre de valeur d'une image
 * \param data données
 * \param buf buffer
 */
void read_mnist_char(char *file, int num_data, int len_info, int n, unsigned char **data, unsigned int *buf)
{
  int i, fd;
  unsigned char *ptr;

  if ((fd = open(file, O_RDONLY)) == -1)
  {
    fprintf(stderr, "couldn't open image file");
    exit(-1);
  }

  read(fd, buf, len_info * sizeof(int));

  for (i = 0; i < len_info; i++)
  {
    ptr = (unsigned char *)(buf + i);
    fliplong(ptr);
    ptr = ptr + sizeof(int);
  }

  for (i = 0; i < num_data; i++)
  {
    read(fd, data[i], n * sizeof(unsigned char));
  }

  close(fd);
}

/**
 * Transformer les valeurs des images de type char en double.
 * \param num_data nombre de données
 * \param data_image_char valeurs des images (type char)
 * \param data_image valeurs des images (type double)
 */
void image_char2double(int num_data, unsigned char **data_image_char, double **data_image)
{
  int i, j;
  for (i = 0; i < num_data; i++)
    for (j = 0; j < MNIST_SIZE; j++)
      data_image[i][j] = ((double)data_image_char[i][j] - 127.5) / 127.5;
}

/**
 * Transformer les labels de type char en int.
 * \param num_data nombre de données
 * \param data_label_char label (type char)
 * \param data_label label (type int)
 */
void label_char2int(int num_data, unsigned char **data_label_char, unsigned int *data_label)
{
  int i;
  for (i = 0; i < num_data; i++)
    data_label[i] = (int)data_label_char[i][0];
}

/**
 * Afficher les valeurs d'une image.
 * \param num_data nombre de données
 * \param data_image valeurs de l'image
 */
void print_data(int num_data, double **data_image)
{
  int i, j;
  for (i = 0; i < num_data; i++)
    for (j = 0; j < MNIST_SIZE; j++)
      printf("%f\n", data_image[i][j]);
}

/**
 * Initialise les paramètres pour la structure
 * mnist_t.
 * 
 * \param output_file: le fichier de sortie
 * \return structure mnist_t 
 */
mnist_t *init_mnist(char *output_file)
{
  int i;
  mnist_t *mnist = (mnist_t *)malloc(sizeof(*mnist));
  assert(mnist);

  unsigned int *info_image = (unsigned int *)malloc(MNIST_LEN_INFO_IMAGE * sizeof(*info_image));
  assert(info_image);
  unsigned int *info_label = (unsigned int *)malloc(MNIST_LEN_INFO_LABEL * sizeof(*info_label));
  assert(info_label);
  unsigned int *train_label = (unsigned int *)malloc(MNIST_NUM_TRAIN * sizeof(*train_label));
  assert(train_label);

  double **train_image = (double **)malloc(MNIST_NUM_TRAIN * sizeof(*train_image));
  assert(train_image);
  for (i = 0; i < MNIST_NUM_TRAIN; i++)
  {
    train_image[i] = (double *)malloc(MNIST_SIZE * sizeof(*train_image[i]));
    assert(train_image[i]);
  }

  unsigned char **image = (unsigned char **)malloc(MNIST_MAX_IMAGESIZE * sizeof(*image));
  assert(image);

  for (i = 0; i < MNIST_MAX_IMAGESIZE; i++)
  {
    image[i] = (unsigned char *)malloc(MNIST_MAX_IMAGESIZE * sizeof(*image[i]));
    assert(image[i]);
  }

  unsigned char **train_image_char = (unsigned char **)malloc(MNIST_NUM_TRAIN * sizeof(*train_image_char));
  assert(train_image_char);

  for (i = 0; i < MNIST_NUM_TRAIN; i++)
  {
    train_image_char[i] = (unsigned char *)malloc(MNIST_SIZE * sizeof(*train_image_char[i]));
    assert(train_image_char[i]);
  }

  unsigned char **train_label_char = (unsigned char **)malloc(MNIST_NUM_TRAIN * sizeof(*train_label_char));
  assert(train_label_char);

  for (i = 0; i < MNIST_NUM_TRAIN; i++)
  {
    train_label_char[i] = (unsigned char *)malloc(sizeof(*train_label_char[i]));
    assert(train_label_char);
  }

  mnist->info_image = info_image;
  mnist->info_label = info_label;
  mnist->train_label = train_label;
  mnist->train_image = train_image;
  mnist->image = image;
  mnist->train_image_char = train_image_char;
  mnist->train_label_char = train_label_char;
  mnist->output = output_file;

  return mnist;
}

/**
 * Charge les données MNIST (données d'apprentissage).
 * 
 * \param output_file fichier de sortie
 * \return structure mnist_t
 */
mnist_t *load_mnist(char *output_file)
{
  mnist_t *mnist = init_mnist(output_file);

  char mnist_train_image[] = MNIST_TRAIN_IMAGE;
  char mnist_train_label[] = MNIST_TRAIN_LABEL;

  read_mnist_char(
      mnist_train_image,
      MNIST_NUM_TRAIN,
      MNIST_LEN_INFO_IMAGE,
      MNIST_SIZE,
      mnist->train_image_char,
      mnist->info_image);

  image_char2double(
      MNIST_NUM_TRAIN,
      mnist->train_image_char,
      mnist->train_image);

  read_mnist_char(
      mnist_train_label,
      MNIST_NUM_TRAIN,
      MNIST_LEN_INFO_LABEL,
      1,
      mnist->train_label_char,
      mnist->info_label);

  label_char2int(
      MNIST_NUM_TRAIN,
      mnist->train_label_char,
      mnist->train_label);

  // Retirer une fois les données chargées et castées
  if (mnist->train_image_char)
  {
    free(mnist->train_image_char);
    mnist->train_image_char = NULL;
  }

  if (mnist->train_label_char)
  {
    free(mnist->train_label_char);
    mnist->train_label_char = NULL;
  }
  return mnist;
}

/**
 * Sauvegarder une image en créant un nouveau fichier.
 * \param mnist structure mnist
 */
void save_image(mnist_t *mnist)
{
  char file_name[MNIST_MAX_FILENAME];
  FILE *fp;
  int x, y;

  strcpy(file_name, mnist->output);

  if ((fp = fopen(file_name, "wb")) == NULL)
  {
    fprintf(stderr, "Error: could not open file. \n");
    exit(1);
  }

  fputs("P5\n", fp);
  fputs("# Created by Image Processing\n", fp);
  fprintf(fp, "%d %d\n", MNIST_WIDTH, MNIST_HEIGHT);
  fprintf(fp, "%d\n", MNIST_MAX_BRIGHTNESS);

  for (y = 0; y < MNIST_HEIGHT; y++)
    for (x = 0; x < MNIST_WIDTH; x++)
      fputc(mnist->image[x][y], fp);
  fclose(fp);

  printf("Image was saved successfully. \n");
}

/**
 * Sauvegarde l'image en enregistrant les valeurs
 * de la matrice passée en paramètre dans les valeurs
 * de sortie.
 * \param data_image données de l'image
 * \param mnist structure mnist
 */
void save_mnist_pgm_mat(matrix_t *data_image, mnist_t *mnist)
{
  int x, y;

  for (y = 0; y < MNIST_HEIGHT; y++)
    for (x = 0; x < MNIST_WIDTH; x++)
      mnist->image[x][y] = data_image->data[0][y * MNIST_WIDTH + x] * 255.0;

  save_image(mnist);
}

/**
 * Libère la mémoire de la structure mnist.
 * \param mnist structure mnist
 */
void free_mnist(mnist_t *mnist)
{
  if (mnist->image)
  {
    free(mnist->image);
    mnist->image = NULL;
  }

  if (mnist->info_image)
  {
    free(mnist->info_image);
    mnist->info_image = NULL;
  }

  if (mnist->info_label)
  {
    free(mnist->info_label);
    mnist->info_label = NULL;
  }

  if (mnist->train_image)
  {
    free(mnist->train_image);
    mnist->train_image = NULL;
  }

  if (mnist->train_image_char)
  {
    free(mnist->train_image_char);
    mnist->train_image_char = NULL;
  }

  if (mnist->train_label)
  {
    free(mnist->train_label);
    mnist->train_label = NULL;
  }

  if (mnist->train_label_char)
  {
    free(mnist->train_label_char);
    mnist->train_label_char = NULL;
  }
}