# SOM

Le Self-Organizing Map [Koh01], également connu sous le nom de SOM, ou encore les cartes de Kohonen, est un réseau de neurones qui s’appuie sur des méthodes d’apprentissage non supervisées. Les cartes de Kohonen se basent sur des cartes qui permettent de réaliser le clustering des données et donc d’étudier la répartition des données selon leurs caractéristiques : c’est ainsi que nous réaliserons la classifications des données textuelles entrées par les utilisateurs.

# Usage
- lancer le programme de façon brute : ``` make && ./som iris_data ```
- lancer le programme plusieurs fois : ``` make test ```
- archiver le programme : ``` make dist ```
- exécuter la documentation du programme via Doxyfile: ``` make doc ```
A noter que le programme s'exécute avec les flags ``` -Wall ``` et ``` -O3 ``` afin d'éviter les Warnings et de réduire le temps d'éxécution.

# TODO List
- fixer la dimension de la grille à l'aide de la formule suivante : ```5 * data_sz ^ 0.5``` où ```data_sz```correspond à la taille de base de données
- écrire la fonction de voisinage
- construire GSOM
- implémenter un système de conscience lors de la phase d'apprentissage