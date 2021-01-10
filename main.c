#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "config.h"
#include "mnist.h"
#include "matrix.h"
#include "gan.h"

#define CONFIG_FILENAME "gan.cfg"

int main(void)
{
  srand(time(NULL));

  mnist_t *mnist = load_mnist();

  const char config_file[] = CONFIG_FILENAME;
  config_t *cfg = init_config(config_file);
  load_mnist_config(cfg, mnist);

  gan_t *gan = init_gan(cfg);

  train_gan(cfg, gan, mnist);
  return 0;
}