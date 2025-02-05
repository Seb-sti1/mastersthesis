Collaborative navigation in unstructured environments using an aerial drone and a terrestrial robot
===

## Initial description

This project explores the collaboration between an aerial drone and a terrestrial robot for navigating unstructured
environments. The drone performs initial environmental mapping using sensors (e.g. GNSS, cameras...), generating a
topological map with identified paths, intersections, and potential targets. The terrestrial robot receives high-level
instructions (e.g., “reach a target”) and navigate using this topological map. It utilizes onboard sensors to match
environmental features previously detected with the drone and associated to the topological map.

## Build latex document

```sh
docker build -t sebsti1/latex
docker run -v /path/to/mastersthesis:/app/ --name latex sebsti1/latex make
```

## Acknowledgements

This is the source files of my Master's thesis for my M.Sc. Eng in Autonomous Systems
at the [Danmarks Tekniske Universitet](https://www.dtu.dk/english/).

It was supervised by [Søren HANSEN](https://orbit.dtu.dk/en/persons/s%C3%B8ren-hansen) and co-supervised
by [Alexandre CHAPOUTOT](https://perso.ensta-paris.fr/~chapoutot/)
and [Thibault TORALBA](http://u2is.ensta-paris.fr/members/toralba/index.php?lang=fr).