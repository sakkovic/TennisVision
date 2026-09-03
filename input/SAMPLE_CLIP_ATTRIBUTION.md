# Attribution for `sample_serve.mp4`

This clip is bundled only so the pipeline can be run and verified before you
supply your own footage. It is not part of the product.

| | |
|---|---|
| **Title** | Maria Sharapova on Amelia Island |
| **Source** | Wikimedia Commons — https://commons.wikimedia.org/wiki/File:Maria_Sharapova_on_Amelia_Island.ogv |
| **Licence** | Creative Commons Attribution 2.0 Generic (CC BY 2.0) — https://creativecommons.org/licenses/by/2.0/ |
| **Changes made** | Trimmed to a 120-frame (4.0 s) excerpt and re-encoded from Ogg Theora to H.264 MP4. Frame size and rate are unchanged at 500x374, 29.97 fps. |

The CC BY 2.0 licence permits this reuse and redistribution, including with
modification, provided the source and licence are credited and the changes are
indicated, which this file does.

The clip shows a **serve**, not a forehand. It is used because it is an openly
licensed clip of a single, fully visible tennis player, which is what the
computer-vision pipeline needs in order to be exercised. Replace it with your own
`forehand.mp4` for real analysis.

To remove it, delete `input/sample_serve.mp4`. Nothing in the codebase depends on
it; the test suite skips the tests that use it when it is absent.
