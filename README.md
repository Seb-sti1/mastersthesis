Collaborative navigation in unstructured environments using an aerial drone and a terrestrial robot
===

## Initial description

This project explores the collaboration between an aerial drone and a terrestrial robot for navigating unstructured
environments. The drone performs initial environmental mapping using sensors (e.g. GNSS, cameras...), generating a
topological map with identified paths, intersections, and potential targets. The terrestrial robot receives high-level
instructions (e.g., “reach a target”) and navigate using this topological map. It utilizes onboard sensors to match
environmental features previously detected with the drone and associated to the topological map.

## Content

- [Bibliography](bibliography): Currently containing notes on the articles that I have read (will be refactored).
- [Datasets](datasets): An empty folder where the datasets should be stored. The list and links to the datasets used are
  specified in the [README.md](datasets/README.md).
- [Latex](latex): The latex source files for all the related work of the Master's thesis.
- [Scripts](scripts): Scripts to test/automate things. If some script become more than a test, it will be moved to a
  dedicated repository (the list will be made available below).

## Build latex & run scripts

### Build latex

```sh
docker build -t sebsti1/latex
docker run -v /path/to/mastersthesis:/app/latex --name latex sebsti1/latex make
```

### Run scripts

#### Env installation

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

#### Run scripts

```sh
# (after sourcing the virtual environment)
python -m scripts.a_script
```

## Acknowledgements

This is the source files of my Master's thesis for my M.Sc. Eng in Autonomous Systems
at the [Danmarks Tekniske Universitet](https://www.dtu.dk/english/). I took place at
the [U2IS lab of ENSTA Paris](http://u2is.ensta-paris.fr/?lang=fr).

It was supervised by [Søren HANSEN](https://orbit.dtu.dk/en/persons/s%C3%B8ren-hansen) and co-supervised
by [Alexandre CHAPOUTOT](https://perso.ensta-paris.fr/~chapoutot/)
and [Thibault TORALBA](http://u2is.ensta-paris.fr/members/toralba/index.php?lang=fr).

## License

All the code (scripts, latex, etc.) is under [GNU GPL](LICENSE) license.
The graph, datasets, etc. are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
