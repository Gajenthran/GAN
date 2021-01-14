#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "mnist.h"
#include "matrix.h"
#include "gan.h"
#define CONFIG_FILENAME "gan.cfg"

/**
 * Cas d'usage de notre programme.
 */
void usage(char* exec)
{
  fprintf(stderr, "Usage: %s<output_filename> \n", exec);
  exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 2)
    usage(argv[0]);

  srand(time(NULL));

  mnist_t* mnist = load_mnist(argv[1]);

  const char config_file[] = CONFIG_FILENAME;
  config_t* cfg = init_config(config_file);
  load_mnist_config(cfg, mnist);

  gan_t* gan = init_gan(cfg);
  train_gan(cfg, gan, mnist);
  save_mnist_pgm_mat(gan->g->a[gan->nb_layers - 2], mnist);

  return 0;
}