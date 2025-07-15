# Notes

## Research papers

The exported data from Zotero are available in the following format:

- [GitHub readable](Master's%20Thesis.html)
- [Zotero importable](Master's%20Thesis.rdf)
- [BibTex](../latex/appendices/mastersthesis.bib)

## Results & Slides

Results are in [results.md](results.md) and slides for various presentations are in the [slides](slides/) folder.

The slides use [marp-cli](https://github.com/marp-team/marp-cli) (especially
the [docker image](https://hub.docker.com/r/marpteam/marp-cli/)) to create
presentations using markdown. For instance, here is the command to create the traversability slides
`docker run --rm --init -v $PWD:/home/marp/app/ -e LANG=$LANG -e MARP_USER="$(id -u):$(id -g)" marpteam/marp-cli notes/slides/traversability/slides.md --pdf --allow-local-files`

## Useful links

- UAV Datasets
    - https://github.com/OpenDroneMap/ODMdata
        - https://hub.dronedb.app/r/odm/aukerman
        - https://github.com/zivillian/odm_ziegeleipark/tree/master
        - https://drive.google.com/file/d/12UrLDHA6iZFJYF7OkErccH_xanRHefoZ/view
        - https://drive.google.com/file/d/1faBtGK7Jm5lTo_UWLz6onDGYGzqlykHPa/view
        - https://drive.google.com/file/d/11yFommuRZyVXADcYEIjf-qz3AZrA8M6E/view
    - https://arxiv.org/pdf/2501.18351 (fpv)
- Camp de Beynes:
    - Geoportail :
      https://remonterletemps.ign.fr/telecharger/?lon=1.889434&lat=48.860337&z=13.4&layer=pva&couleur=C&year=2013
    - Google Maps : https://maps.app.goo.gl/zCesePWSyXjnszSf7
- Similarity of images
    - https://github.com/lbrejon/Compute-similarity-between-images-using-CNN?tab=readme-ov-file
    - https://pubs.aip.org/aip/acp/article-abstract/3092/1/040015/3270019/Similar-image-retrieval-using-convolutional-neural?redirectedFrom=fulltext
    - https://deeplobe.ai/image-similarity-using-deep-cnn-theory-to-code/
    - https://medium.com/@f.a.reid/image-similarity-using-feature-embeddings-357dc01514f8
- Meshroom https://github.com/alicevision/Meshroom
- Labelme https://github.com/wkentaro/labelme
- marp-cli (markdown slides) https://github.com/marp-team/marp-cli