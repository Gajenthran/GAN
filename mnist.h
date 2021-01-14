/*!
 * \file mnist.h
 * \brief Fichier header de mnist.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __MNIST_H__
#define __MNIST_H__

#include "matrix.h"
#include "mnist.h"

// Fichier pour les données d'apprentissage MNIST
#define MNIST_TRAIN_IMAGE "./data/train-images.idx3-ubyte"
// Fichier pour les labels MNIST
#define MNIST_TRAIN_LABEL "./data/train-labels.idx1-ubyte"

// Taille d'une image MNIST (28 * 28)
#define MNIST_SIZE 784
// Taille de la longueur d'une image MNIST
#define MNIST_WIDTH 28
// Taille de la largeur d'une image MNIST
#define MNIST_HEIGHT 28
// Nombre de données d'apprentissage
#define MNIST_NUM_TRAIN 60000
// Taille pour les informations sur l'image pour le buffer
#define MNIST_LEN_INFO_IMAGE 4
// Taille pour les informations sur le label pour le buffer
#define MNIST_LEN_INFO_LABEL 2
// Taille max. pour les images
#define MNIST_MAX_IMAGESIZE 1280
// Luminosité max. pour les images
#define MNIST_MAX_BRIGHTNESS 255
// Taille max. pour le nom des images
#define MNIST_MAX_FILENAME 256
// Nombre max. d'images à stocker
#define MNIST_MAX_NUM_OF_IMAGES 1

typedef struct mnist mnist_t;
/* Structure représentant les données MNIST */
struct mnist {
  unsigned int* train_label; // labels MNIST
  unsigned int* info_image; // informations sur l'image pour le buffer
  unsigned int* info_label; // informations sur le label pour le buffer
  double** train_image; // données d'apprentissage MNIST
  unsigned char** image; // l'image en sortie (sauvegardée)
  unsigned char** train_image_char; // données d'apprentissage MNIST brutes
  unsigned char** train_label_char; // labels MNIST brutes
  char* output; // nom du fichier en sortie (de l'image sauvegardé)
};

mnist_t* init_mnist(char*);
mnist_t* load_mnist(char*);
void read_mnist_char(char*, int, int, int, unsigned char**, unsigned int*);
void image_char2double(int, unsigned char**, double**);
void label_char2int(int, unsigned char**, unsigned int*);
void save_image(mnist_t*);
void save_mnist_pgm_mat(matrix_t*, mnist_t*);

#endif