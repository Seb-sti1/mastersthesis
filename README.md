Collaborative navigation in unstructured environments using an aerial drone and a terrestrial robot
===

> [!IMPORTANT]
> To clone, it is recommended to install `git-lfs` on your system.

## Initial description

This project explores the collaboration between an aerial drone and a terrestrial robot for navigating unstructured
environments. The drone performs initial environmental mapping using sensors (e.g. GNSS, cameras...), generating a
topological map with identified paths, intersections, and potential targets. The terrestrial robot receives high-level
instructions (e.g., “reach a target”) and navigate using this topological map. It utilizes onboard sensors to match
environmental features previously detected with the drone and associated to the topological map.

## Content

- [Datasets](datasets): An empty folder where the datasets should be stored. The list and links to the datasets used are
  specified in the dedicated [README.md](datasets/README.md).
- [Docker](docker): Dockerfiles to simplify reproducibility.
- [Latex](latex): The latex source files for all the related work of the Master's thesis.
- [Notes](notes): The notes of my research and intermediate results. See dedicated [README.md](notes/README.md).
- [Scripts](scripts): Scripts to test/automate things. If some script become more than a test, it will be moved to a
  dedicated repository (the list will be made available below).

## Build latex & run scripts

See the [docker/README.md](docker/README.md) for more information about the dependencies
and [scripts/README.md](scripts/README.md) for the scripts.

## Acknowledgements

This is the source files of my Master's thesis for my M.Sc. Eng in Autonomous Systems
at the [Danmarks Tekniske Universitet](https://www.dtu.dk/english/). It took place at
the [U2IS lab of ENSTA Paris](http://u2is.ensta-paris.fr/?lang=fr).

It was supervised by [Søren HANSEN](https://orbit.dtu.dk/en/persons/s%C3%B8ren-hansen) and co-supervised
by [Alexandre CHAPOUTOT](https://perso.ensta-paris.fr/~chapoutot/)
and [Thibault TORALBA](http://u2is.ensta-paris.fr/members/toralba/index.php?lang=fr).

## License

All the code (scripts, latex, etc.) is under [GNU GPL](LICENSE) license.
The graph, datasets, images, etc. (excluding items mentioned but not present in this repository) are
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
