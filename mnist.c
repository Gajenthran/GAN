#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include "config.h"

void FlipLong(unsigned char *ptr)
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

void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char **data_char, unsigned int *info_arr)
{
    int i, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1)
    {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }

    read(fd, info_arr, len_info * sizeof(int));

    // read-in information about size of data
    for (i = 0; i < len_info; i++)
    {
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
        ptr = ptr + sizeof(int);
    }

    // read-in mnist numbers (pixels|labels)
    for (i = 0; i < num_data; i++)
    {
        read(fd, data_char[i], arr_n * sizeof(unsigned char));
    }

    close(fd);
}

void image_char2double(int num_data, unsigned char **data_image_char, double **data_image)
{
    int i, j;
    for (i = 0; i < num_data; i++)
        for (j = 0; j < MNIST_SIZE; j++)
            data_image[i][j] = ((double)data_image_char[i][j] - 127.5) / 127.5;
}

void label_char2int(int num_data, unsigned char **data_label_char, unsigned int *data_label)
{
    int i;
    for (i = 0; i < num_data; i++)
        data_label[i] = (int)data_label_char[i][0];
}

void print_data(int num_data, double **data_image)
{
    int i, j;
    for (i = 0; i < num_data; i++)
        for (j = 0; j < MNIST_SIZE; j++)
            printf("%f\n", data_image[i][j]);
}

mnist_t *init_mnist(void)
{
    int i;
    mnist_t *mnist = (mnist_t *)malloc(sizeof(*mnist));
    assert(mnist);

    unsigned int *info_image = (unsigned int *)malloc(MNIST_LEN_INFO_IMAGE * sizeof(*info_image));
    unsigned int *info_label = (unsigned int *)malloc(MNIST_LEN_INFO_LABEL * sizeof(*info_label));
    unsigned int *train_label = (unsigned int *)malloc(MNIST_NUM_TRAIN * sizeof(*train_label));

    double **train_image = (double **)malloc(MNIST_NUM_TRAIN * sizeof(*train_image));
    for (i = 0; i < MNIST_NUM_TRAIN; i++)
        train_image[i] = (double *)malloc(MNIST_SIZE * sizeof(*train_image[i]));

    unsigned char **image = (unsigned char **)malloc(MNIST_MAX_IMAGESIZE * sizeof(*image));
    for (i = 0; i < MNIST_MAX_IMAGESIZE; i++)
        image[i] = (unsigned char *)malloc(MNIST_MAX_IMAGESIZE * sizeof(*image[i]));

    unsigned char **train_image_char = (unsigned char **)malloc(MNIST_NUM_TRAIN * sizeof(*train_image_char));
    for (i = 0; i < MNIST_NUM_TRAIN; i++)
        train_image_char[i] = (unsigned char *)malloc(MNIST_SIZE * sizeof(*train_image_char[i]));

    unsigned char **train_label_char = (unsigned char **)malloc(MNIST_NUM_TRAIN * sizeof(*train_label_char));
    for (i = 0; i < MNIST_NUM_TRAIN; i++)
        train_label_char[i] = (unsigned char *)malloc(sizeof(*train_label_char[i]));

    mnist->info_image = info_image;
    mnist->info_label = info_label;
    mnist->train_label = train_label;
    mnist->train_image = train_image;
    mnist->image = image;
    mnist->train_image_char = train_image_char;
    mnist->train_label_char = train_label_char;

    return mnist;
}

mnist_t *load_mnist(void)
{
    mnist_t *mnist = init_mnist();

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

    return mnist;
}

// name: path for saving image (ex: "./images/sample.pgm")
void save_image(char name[], mnist_t *mnist)
{
    char file_name[MNIST_MAX_FILENAME];
    FILE *fp;
    int x, y;

    if (name[0] == '\0')
    {
        printf("output file name (*.pgm) : ");
        scanf("%s", file_name);
    }
    else
        strcpy(file_name, name);

    if ((fp = fopen(file_name, "wb")) == NULL)
    {
        printf("could not open file\n");
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
    printf("Image was saved successfully\n");
}

// save mnist image (call for each image)
// store train_image[][] into image[][][]
void save_mnist_pgm_mat(matrix_t *data_image, mnist_t *mnist)
{
    char filename[] = "./toto.png";
    int x, y;

    for (y = 0; y < MNIST_HEIGHT; y++)
        for (x = 0; x < MNIST_WIDTH; x++)
            mnist->image[x][y] = data_image->data[0][y * MNIST_WIDTH + x] * 255.0;

    save_image(filename, mnist);
}
