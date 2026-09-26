# ISRO entity images

`isro_matcher.py` returns each entity's `image` field verbatim (see
`src/isro_kb.json`); the UI resolves it against `/assets/isro/<filename>`.

An entity with `"image": null` (or whose file fails to load) gets a styled
panel in the mission card -- an orbit emblem with the mission name and
status -- instead of a picture, so a missing photo never looks broken.

To add a real image: drop the file in this directory and set that entity's
`image` field in `isro_kb.json` to the filename. The auto-generated
`*.svg` placeholders in this directory are no longer referenced by the KB
(they printed "placeholder image" on screen); they can be deleted.

## Real images (in progress)

7 entities point at real photos: `chandrayaan-3`, `gaganyaan`, `mangalyaan`,
`pslv`, `gslv-mk3` (file `lvm3.jpg`), `navic`, `vikram-lander`. The other 13
have `"image": null` pending sourcing -- see the list below.

- Untouched originals live in `originals/` (never served, not resized).
  The working copies in this directory are resized to a max 1600px on the
  long edge and JPEG-compressed under ~200KB each, aspect ratio preserved
  (several are tall portrait shots -- do not crop them to landscape in the
  UI; let the card adapt to the image).
- `isro-logo.png` is the UI header logo (transparent background) -- kept
  as PNG, not resized/recompressed like the entity photos.
- `lvm3-pad.jpg` (LVM3-M6 on the pad at night) is kept in this directory as
  an unused alternate/backup shot for `gslv-mk3` -- not referenced by
  `isro_kb.json` today.
- **`navic.jpg` carries a third-party watermark (trackobit.com)** in the
  corner -- it's a placeholder-grade stand-in for the real constellation
  diagram, not final. Swap it for an ISRO-sourced graphic before
  submission.
<!-- FILENAME_LIST -->
- **needs image** -- Chandrayaan-1 (`chandrayaan-1`)
- **needs image** -- Chandrayaan-2 (`chandrayaan-2`)
- `chandrayaan-3.jpg` -- Chandrayaan-3
- `gaganyaan.jpg` -- Gaganyaan
- `mangalyaan.jpg` -- Mangalyaan (Mars Orbiter Mission)
- **needs image** -- Aditya-L1 (`aditya-l1`)
- `pslv.jpg` -- PSLV
- `lvm3.jpg` -- GSLV Mk-III / LVM3
- `navic.jpg` -- NavIC
- **needs image** -- RISAT (`risat`)
- `vikram-lander.jpg` -- Vikram lander
- **needs image** -- Pragyan rover (`pragyan-rover`)
- **needs image** -- SDSC Sriharikota (`sdsc-sriharikota`)
- **needs image** -- VSSC Thiruvananthapuram (`vssc-thiruvananthapuram`)
- **needs image** -- ISTRAC Bengaluru (`istrac-bengaluru`)
- **needs image** -- NISAR (`nisar`)
- **needs image** -- Aryabhata (`aryabhata`)
- **needs image** -- INSAT (`insat`)
- **needs image** -- Cartosat (`cartosat`)
- **needs image** -- Shukrayaan (`shukrayaan`)
