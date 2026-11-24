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
3. After arXiv assigns an ID, add the arXiv abstract URL to the paper actions in
   `index.html` and update the BibTeX there and in the root `README.md`. The
   current **Read paper** link opens the bundled preprint PDF.

The social preview image is `assets/preview.jpg`. Update the `og:image` URL in
`index.html` if the site uses a different host or repository name.

Geist and Instrument Serif are distributed under the SIL Open Font License.
The inline icons are from Lucide. Their licenses are in `assets/`.
