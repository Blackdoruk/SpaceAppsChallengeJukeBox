# Earth Information Jukebox

NASA Space Apps Challenge Ankara 2026 — code and sound-generation guide.

This project translates features in NASA Earth visualizations into synchronized sound. A visible scanline connects the region being analyzed to the sound being heard. The developed examples cover atmospheric aerosols and lightning activity.

The current Python pipelines render audio and video files offline. They analyze the colors and shapes in rendered imagery, rather than reading the underlying scientific measurements directly. The Figma interface presents the intended Earth exploration experience. Its Maps and Studio screens are a prototype, and the Python code does not establish a working connection to that interface.

## What we used

| Tool | Role in the project |
| --- | --- |
| Python and Jupyter notebooks | Configure and run the image analysis and audio rendering workflows. |
| OpenCV (`cv2`) | Decode video, resize frames, detect the globe, analyze HSV colors, detect connected yellow regions, and draw the scanline. |
| NumPy | Calculate color weights, smooth volume changes, construct audio timelines, and mix samples. |
| FFmpeg | Decode WAV/MP3 files into sample arrays, process the lightning sound, and encode the final H.264/AAC MP4. |
| `imageio-ffmpeg` | Supply an FFmpeg executable when one is unavailable on PATH. |
| Python `wave` module | Save 44.1 kHz, stereo, 16-bit PCM WAV files. |
| `requests` | Query NASA SVS APIs in the data-discovery notebook. |
| Pixasonics | Explore mapping image features to synthesized sound in the experimental notebooks. |
| Figma | Prototype the map-layer selection and proposed Studio workflow. |

The current aerosol and lightning renderers use audio samples already supplied as WAV/MP3 files. They do not require Pixasonics or SuperCollider. The workspace contains a SuperCollider source tree and an initialization log, but these do not establish that SuperCollider created the supplied samples. The code does not record the original recording tools, authors, or licenses of those samples.

## Code guide

| File | Purpose |
| --- | --- |
| `color_sampled_pixasonics.ipynb` | Current aerosol sonification using recorded samples and a moving scanline. Despite its filename, this renderer does not import Pixasonics. |
| `lightning_sonification.py` | Current lightning detection, hit construction, event scheduling, and audiovisual export. |
| `lightning_sonification.ipynb` | Notebook companion to the lightning script. Check the script for the latest audio implementation. |
| `nasa_eic_data_collect.ipynb` | Search NASA SVS for Earth Information Center content and print discovered video URLs. It does not download the videos. |
| `pixasonics_test.ipynb` | Map the mean red channel of a still image to a Theremin synth's frequency with exponent 2. |
| `pixasonics_temporal_demo.ipynb` | Feed successive video frames into Pixasonics and update the same image-to-frequency mapping. This is a timing experiment, not a production streaming player. |
| `color_sampled_pixasonics_original.ipynb` | Earlier sample-mixing experiment using image sequences, color layers, and white-region kick triggers. Its input paths differ from the current aerosol notebook. |

## Aerosols: colors control a continuous sound mix

**Input video:** `../filteredData/Aerosols/Aerosols_RotatingEarth_3840x2160.mp4`

The notebook makes one top-to-bottom scan over the video duration. It estimates the Earth disk from the first frame and confines the white scanline overlay to that disk. For each frame, it analyzes the corresponding horizontal row in HSV color space.

| Visual feature | Audio file | Configured gain |
| --- | --- | --- |
| Continuous background | `arkaplan.wav` | 0.4 |
| Green | `yeşil.wav` | 2.0 |
| Blue | `mavi.mp3` | 2.0 |
| Purple | `mor.wav` | 2.0 |
| Red, background layer | `red_background.mp3` | 0.7 |
| Red, foreground layer | `red_foreground.mp3` | 1.0 |

All files above live in `../filteredData/Aerosols/`. These gains multiply the prepared samples and their color envelopes. They are not percentages or scientific units.

### How the sound is constructed

1. FFmpeg decodes each sound into 44.1 kHz stereo floating-point samples. The decoder level-matches each source toward -24 dBFS RMS, subject to a peak limit.
2. Samples repeat across the timeline. Crossfading the loop boundary reduces clicks. The red background sample excludes its first 8 seconds.
3. For each video frame, the code measures circular hue distance from each configured color. A triangular membership function lets nearby hues contribute gradually. Saturation and brightness suppress weak or dark pixels.
4. The mean membership across the scan row determines the target volume of each sound layer. More matching pixels generally make that layer louder. The current mean uses the entire image row, including background pixels. The globe mask clips the overlay, but does not crop the color-weight calculation.
5. Exponential envelopes smooth changes over 180 ms by default. Both red layers use 700 ms smoothing and 500 ms loop crossfades.
6. NumPy adds the weighted layers to the continuous background. The final mix targets -18 dBFS RMS with a 0.95 peak ceiling. The peak ceiling can prevent it from reaching the RMS target.
7. The code saves a WAV and uses FFmpeg to combine it with the scanline video.

This is amplitude/RMS normalization, not LUFS normalization. The mapping controls the loudness and mixture of supplied sounds, rather than generating new notes from physical aerosol concentrations. Color meanings must be checked against the original NASA visualization legend before assigning scientific labels to them.

**Outputs in `outputs/`:**

- `aerosols_sonification.mp4`: final video with audio.
- `aerosols_sonification.wav`: rendered audio.
- `aerosols_scan_silent.mp4`: intermediate video with the scanline.
- `scan_check_*.png`: frames saved for visual inspection.

## Lightning: detected dots trigger individual hits

**Input video:** `../filteredData/Lightning/iss_lightning_sphere_07312023_60p4k.mp4`

The current script starts at 85 seconds into the source and makes four top-to-bottom sweeps over the remaining segment. It detects the Earth disk using a Hough circle transform. Inside the disk, HSV thresholds isolate yellow pixels, and connected-component analysis groups them into visible dots.

When a dot crosses the scanline, the script schedules one composite hit. It tracks previously triggered regions between adjacent frames using a dilated mask to reduce duplicate triggers as the globe rotates. Tracking resets at each sweep. Distinct dots crossing together each produce a hit.

### What produces the lightning sound

The latest script layers two files from `../filteredData/Lightning/`:

- **`yellow.mp3`:** the main tonal/bell sample. FFmpeg lowers its pitch through a sample-rate change from 44,100 to 39,289 Hz and resamples it back to 44.1 kHz. This slows and lengthens the decoded sample before trimming. It adds bass around 220 Hz, reduces treble around 3.5 kHz, and applies a 4.5 kHz low-pass filter for a warmer tone.
- **`yellow_kick.mp3`:** a percussive layer mixed at gain 0.7 beneath the yellow sample.

The loader trims leading silence using a 2% peak threshold, with a short margin before the detected onset. It limits the yellow sample to 1.5 seconds and applies a smooth 0.8-second cosine fade-out. It level-matches each source toward -18 dBFS RMS subject to a peak constraint. The kick uses a short default fade-out. The two sounds start together for every trigger.

The renderer schedules hits at sample positions derived from the video frame times. Overlapping hits add together, and the complete mix is attenuated if its peak exceeds 0.95. The source sound files remain unchanged.

The white line indicates the scanning operation. Trigger timing represents when the scan reaches a visible marker, not the original observation time of an individual lightning strike. Dense or merged dots and the approximate tracking method can affect the number of triggers.

**Outputs in `outputs/lightning/`:**

- `lightning_sonification.mp4`: final video with audio.
- `lightning_sonification.wav`: rendered audio.
- `lightning_scan_silent.mp4`: intermediate scanline video.
- `lightning_events.json`: source offset, rendered duration, sweep settings, and trigger times relative to the output segment.
- `scan_check_*.png`: inspection frames.

Existing outputs may reflect an earlier configuration. Re-render after changing settings to make the output match the current script.

## Setup and running

Keep the repository beside `filteredData`, or edit the input paths:

```text
workspace/
  filteredData/
    Aerosols/                 # Source video and six sound files
    Lightning/                # Source video, yellow.mp3, yellow_kick.mp3
  SpaceAppsChallengeJukeBox/
    README.md
    color_sampled_pixasonics.ipynb
    lightning_sonification.py
    lightning_sonification.ipynb
    outputs/
```

Install the packages into your chosen Python environment:

```bash
python -m pip install numpy opencv-python imageio-ffmpeg jupyter requests
```

The scripts also look in the project-local `.notebook-deps` directory. Use a Jupyter kernel with the required packages installed. The lightning script imports `imageio_ffmpeg` even when a system FFmpeg is available, so install that package for both workflows.

### Render aerosols

Open `color_sampled_pixasonics.ipynb` in Jupyter from the repository folder or its parent. Run its cells in order. Set `PREVIEW_SECONDS = 5.0` for a short check, or `None` for the complete video. Modify `COLOR_SOUNDS` to change sample paths, hue ranges, gains, source offsets, or smoothing settings.

### Render lightning

From the repository folder:

```bash
python lightning_sonification.py
```

Alternatively, use the companion notebook. Edit the constants at the top of the script to change the source offset, number of sweeps, sound files, tone filter, or preview duration. Both `yellow.mp3` and `yellow_kick.mp3` are required by the current implementation.

To change the sound without rescanning the video, the script also exposes `remix_existing()`. From the repository folder:

```bash
python -c "import lightning_sonification as ls; ls.remix_existing()"
```

This requires the existing silent video and `lightning_events.json`. It checks the cached source offset and sweep count, then replaces the audio outputs using the cached triggers. It does not validate every input or preview setting, so run a full render if the source video, duration, or detection logic changed.

Both pipelines preserve source frame timing for the selected segment and downscale to a maximum width of 1280 pixels without intentionally changing the aspect ratio. They export H.264 video with AAC audio at 192 kbps, plus a separate PCM WAV. Rendering writes to fixed output names and can replace earlier results. Audio buffers accumulate in memory, so start with a short preview for large inputs.

## Interpretation and current scope

- The outputs offer an auditory representation of visible patterns. They do not reconstruct calibrated measurements from pixel colors.
- No machine learning model is used in the current aerosol or lightning renderers. Their mappings use explicit computer-vision and audio-processing rules.
- The primary renderers use existing samples. The experimental Pixasonics notebooks demonstrate synthesized sound by mapping mean red intensity to Theremin frequency.
- Accessibility is an intended benefit. The repository does not demonstrate screen-reader integration or a completed usability study with blind or low-vision participants.
- Other layers shown in the interface, such as fire, greenhouse gases, and CO₂ emissions, do not by themselves establish completed sonification pipelines.
- Verify sample credits and licenses before public distribution. The supplied code does not document their provenance.

## Sources and prototype

- [NASA Space Apps challenge brief](https://www.spaceappschallenge.org/2026/challenges/the-earth-information-jukebox/)
- [NASA Scientific Visualization Studio](https://svs.gsfc.nasa.gov/)
- [NASA SVS search API used by the discovery notebook](https://svs.gsfc.nasa.gov/api/search/)
- [Figma interface prototype](https://www.figma.com/proto/zTRQbFBp80Mhd6w9oXH552/Untitled?node-id=48-129)

For a presentation, describe the work as **sample-based sonification of NASA Earth visualizations, using color-controlled continuous mixing for aerosols and event-triggered layered hits for lightning**.
