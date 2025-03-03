# Notes

## Research papers

The exported data from Zotero are available in the following format:

- [GitHub readable](Master's%20Thesis.html)
- [Zotero importable](Master's%20Thesis.rdf)
- [BibTex](../latex/appendices/Master%27s%20Thesis.bib)

## Results & Slides

Results are in [results.md](results.md) and slides for various presentations are in the [slides](slides/) folder.

The slides use [marp-cli](https://github.com/marp-team/marp-cli) (especially
the [docker image](https://hub.docker.com/r/marpteam/marp-cli/)) to create
presentations using markdown. For instance, here is the command to create the traversability slides
`docker run --rm --init -v $PWD:/home/marp/app/ -e LANG=$LANG -e MARP_USER="$(id -u):$(id -g)" marpteam/marp-cli notes/slides/traversability/slides.md --pdf --allow-local-files`