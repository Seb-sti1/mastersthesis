Collaborative navigation in unstructured environments using an aerial drone and a terrestrial robot
===

This repository contains most of the overwhelming majority of the work done during my master's
thesis.

> [!IMPORTANT]
> To clone, it is recommended to install `git-lfs` on your system.

## Abstract

This project explores the collaboration between an aerial drone and a terrestrial robot for navigating unstructured environments.
The drone performs initial environmental mapping using sensors (e.g. GNSS, cameras...), generating a topological map with identified paths, intersections, and potential targets.
The terrestrial robot receives high-level instructions (e.g., “reach a target”) and navigate using this topological map.
It utilizes onboard sensors to match environmental features previously detected with the drone and associated to the topological map.

Techniques are proposed to handle the perspective differences between aerial and ground views, including bird's-eye view generation and sparse correspondence matching.
Simulated real-world scenario demonstrates that the proposed system offers a promising foundation for real-world deployment.
This project explores the collaboration between an aerial drone and a terrestrial robot for navigating unstructured
environments. The drone performs initial environmental mapping using sensors (e.g. GNSS, cameras...), generating a
topological map with identified paths, intersections, and potential targets. The terrestrial robot receives high-level
instructions (e.g., “reach a target”) and navigate using this topological map. It utilizes onboard sensors to match
environmental features previously detected with the drone and associated to the topological map.

## Content

- [Datasets](datasets): An empty folder where the datasets should be stored. The list and links to the datasets used are
  specified in the dedicated [README.md](datasets/README.md).
- [Docker](docker): Dockerfiles to simplify reproducibility.
- [Latex](latex): The latex source files for the project plan, report and defence. 
- [Notes](notes): The notes of my research and intermediate results. See dedicated [README.md](notes/README.md).
- [Scripts](scripts): Scripts to test/automate things. If some script become more than a test, it will be moved to a
  dedicated repository (the list will be made available here).

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

Given that this repository contains multiple type of document, two licenses are used:

- The files in [scripts](scripts) and [docker](docker) (mostly python and dockerfile) are under [GNU GPL](LICENSE)
  license.
- The files in [latex](latex) and [notes](notes) (mostly images, markdown, latex files), except for images in
  [logo](latex/logo), are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license. The templates
  are largely based of [DTU's templates](https://gitlab.gbar.dtu.dk/latex/dtutemplates/tree/master/templates/).
- The images in [logo](latex/logo) are copyrighted, belong to their rightful owner and should be used only when
  permitted by law.
- No datasets will be stored in [datasets](datasets), please refer to the license given by the author of the dataset.

If you have any doubt regarding the licensing of part of this repository, please consider submitting an issue.