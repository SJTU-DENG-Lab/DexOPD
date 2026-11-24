# DexOPD project page

Open `index.html` directly in a browser to preview the complete site. The demo
uses pre-generated outputs from the paper, with prompts and generation settings
in `examples.js`. The source outputs were generated at 1024 x 1024. The site
serves WebP copies of those outputs.

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
