# GAN

## Usage

- ` make && ./gan ./out.png `
- ` make libs ` pour la bibliothèque statique

## Données

- MNIST
- 60000 données d'apprentissage
- Labels numérotés de 1 à 9

## Configuration

- Informations sur les paramètres du GAN
- Informations sur les informations MNIST
- Informations sur l'entraînement du GAN
- Utilisation de hashcode pour lier le fichier config à la structure config
- Cf. gan.cfg

## Matrice

- Bibliothèque d'une matrice 
- Tableau 1D
- les ` _ ` à la fin de chaque fonction signifie que les valeurs seront stockés sur le premier paramètre de la fonction

## GAN

- generator / discriminator
- verbose pour afficher à chaque n iteration
- progressbar

### TODO

- amélioration des propagations avants/arrières
- optimisations des boucles

