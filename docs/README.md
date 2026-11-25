# DexOPD project page

Open `index.html` directly in a browser to preview the overview, interactive
method diagrams, and benchmark/analysis charts, or open `examples.html` for
same-noise model comparisons and the multi-capability gallery. Prompts and
generation settings live in `examples.js`, and benchmark/ablation numbers live
in `results-data.js`. Source qualitative outputs were generated at 1024 x 1024
and served as WebP copies.

## Publish with GitHub Pages

1. Push the repository, including this `docs/` directory, to the public
   `SJTU-DENG-Lab/DexOPD` repository.
2. In **Settings > Pages**, choose **Deploy from a branch** and select the
   default branch and **`/docs`**. The default URL is
   `https://sjtu-deng-lab.github.io/DexOPD/`.
3. The hero **arXiv** button links to the abstract page
   (https://arxiv.org/abs/2611.31969), and **Paper PDF** opens the bundled
   preprint in `assets/`. Keep the BibTeX in `index.html`, the root `README.md`,
   and `MODEL_CARD.md` in sync when the arXiv entry is revised.

The social preview image is `assets/preview.jpg`. Update the `og:image` URL in
`index.html` if the site uses a different host or repository name.

Geist and Instrument Serif are distributed under the SIL Open Font License.
The inline icons are from Lucide. Their licenses are in `assets/`.
