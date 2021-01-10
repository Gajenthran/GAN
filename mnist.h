#ifndef __MNIST_H__
#define __MNIST_H__

#define MNIST_TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define MNIST_TRAIN_LABEL "./data/train-labels.idx1-ubyte"

#define MNIST_SIZE 784 // 28*28
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define MNIST_NUM_TRAIN 60000
#define MNIST_LEN_INFO_IMAGE 4
#define MNIST_LEN_INFO_LABEL 2

#define MNIST_MAX_IMAGESIZE 1280
#define MNIST_MAX_BRIGHTNESS 255
#define MNIST_MAX_FILENAME 256
#define MNIST_MAX_NUM_OF_IMAGES 1

#include "matrix.h"
#include "mnist.h"

typedef struct mnist mnist_t;

struct mnist
{
    unsigned int *train_label;
    unsigned int *info_image;
    unsigned int *info_label;
    double **train_image;
    unsigned char **image;
    unsigned char **train_image_char;
    unsigned char **train_label_char;
};

mnist_t *init_mnist(void);
mnist_t *load_mnist(void);
void FlipLong(unsigned char *);
void read_mnist_char(char *, int, int, int, unsigned char **, unsigned int *);
void image_char2double(int, unsigned char **, double **);
void label_char2int(int, unsigned char **, unsigned int *);
void save_image(char *, mnist_t *);
void save_mnist_pgm_mat(matrix_t *, mnist_t *);

#endif