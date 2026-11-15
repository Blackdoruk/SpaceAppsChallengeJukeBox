"""One-shot yellow-dot sonification; run directly or from the companion notebook."""
from pathlib import Path
import sys
import shutil
import subprocess
import wave
import cv2
import numpy as np

NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR / '.notebook-deps'))
import imageio_ffmpeg
FFMPEG = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
DATA_DIR = NOTEBOOK_DIR.parent / 'filteredData' / 'Lightning'
VIDEO_PATH = DATA_DIR / 'iss_lightning_sphere_07312023_60p4k.mp4'
YELLOW_PATH = DATA_DIR / 'yellow.mp3'
KICK_PATH = DATA_DIR / 'yellow_kick.mp3'
YELLOW_HIT_SECONDS = 1.5
KICK_GAIN = 0.7
START_SECONDS = 85.0  # 01:25 into the source video
PREVIEW_SECONDS = None
MAX_VIDEO_WIDTH = 1280
SAMPLE_RATE = 44100
YELLOW_GAIN = 1.0
SCAN_PASSES = 4  # Four top-to-bottom sweeps: 4x the previous scan speed.
BELL_FADEOUT_SECONDS = 0.8
# A slightly lower pitch and warmer EQ give the yellow hit a fuller "tok" tone.
YELLOW_TONE_FILTER = 'aresample=44100,asetrate=39289,aresample=44100,bass=g=4:f=220,treble=g=-4:f=3500,lowpass=f=4500'
OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / 'lightning'


def load_hit(path, gain=1.0, max_seconds=None, fade_seconds=0.01, tone_filter=None):
    filters = ['-af', tone_filter] if tone_filter else []
    decoded = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), *filters,
                              '-f', 'f32le', '-ac', '2', '-ar', str(SAMPLE_RATE), 'pipe:1'],
                             capture_output=True, check=True)
    sample = np.frombuffer(decoded.stdout, dtype='<f4').reshape(-1, 2).copy()
    if not sample.size or not np.isfinite(sample).all():
        raise ValueError('Yellow sample is empty or invalid.')
    # Remove leading silence so the audible onset coincides with the crossing.
    peak = float(np.max(np.abs(sample)))
    onset = np.flatnonzero(np.max(np.abs(sample), axis=1) > peak * 0.02)
    if not onset.size:
        raise ValueError('Yellow sample is silent.')
    sample = sample[max(0, onset[0] - round(0.003 * SAMPLE_RATE)):]
    if max_seconds is not None:
        sample = sample[:max(2, round(max_seconds * SAMPLE_RATE))]
    rms = float(np.sqrt(np.mean(sample.astype(np.float64)**2)))
    sample *= min(10**(-18 / 20) / rms, 0.9 / peak) * gain
    fade_samples = min(len(sample), round(fade_seconds * SAMPLE_RATE))
    if fade_samples >= 2:
        # Smooth cosine fade preserves the bell attack and reaches zero at its end.
        fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
        sample[-fade_samples:] *= fade[:, None]

    return sample


def combined_hit():
    yellow = load_hit(YELLOW_PATH, YELLOW_GAIN, YELLOW_HIT_SECONDS, BELL_FADEOUT_SECONDS, YELLOW_TONE_FILTER)
    kick = load_hit(KICK_PATH, KICK_GAIN)
    hit = np.zeros((max(len(yellow), len(kick)), 2), dtype=np.float32)
    hit[:len(yellow)] += yellow
    hit[:len(kick)] += kick
    return hit


def write_audio_video(sample, events, duration):
    silent = OUTPUT_DIR / "lightning_scan_silent.mp4"
    audio = np.zeros((round(duration * SAMPLE_RATE), 2), dtype=np.float32)
    for start in events:
        end = min(start + len(sample), len(audio))
        audio[start:end] += sample[:end-start]
    peak = float(np.max(np.abs(audio)))
    if peak > 0.95:
        audio *= 0.95 / peak
    wav_path = OUTPUT_DIR / 'lightning_sonification.wav'
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
    output = OUTPUT_DIR / 'lightning_sonification.mp4'
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-i', str(silent), '-i', str(wav_path),
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                    '-shortest', '-movflags', '+faststart', str(output)], check=True)
    return output


def render():
    if not isinstance(SCAN_PASSES, int) or SCAN_PASSES < 1:
        raise ValueError('SCAN_PASSES must be a positive integer.')
    if BELL_FADEOUT_SECONDS < 0:
        raise ValueError('BELL_FADEOUT_SECONDS must be nonnegative.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (VIDEO_PATH, YELLOW_PATH, KICK_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    sample = combined_hit()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    events = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0:
            raise RuntimeError('Could not open the input video.')
        first = round(START_SECONDS * fps)
        count = total - first
        if count <= 0 or first < 0:
            raise ValueError('START_SECONDS is outside the video.')
        if PREVIEW_SECONDS is not None:
            count = min(count, max(1, round(PREVIEW_SECONDS * fps)))
        width, height = cap.get(3), cap.get(4)
        scale = min(1, MAX_VIDEO_WIDTH / width)
        size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        ok, reference = cap.read()
        if not ok:
            raise RuntimeError('Could not read the starting frame.')
        reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (9, 9), 2)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, size[1] // 2,
                                  param1=80, param2=35,
                                  minRadius=int(size[1] * 0.35), maxRadius=int(size[1] * 0.49))
        if circles is None:
            raise RuntimeError('Earth circle could not be detected; inspect the starting frame.')
        cx, cy, radius = min(circles[0], key=lambda c: (c[0]-size[0]/2)**2+(c[1]-size[1]/2)**2)
        radius -= 2
        yy, xx = np.ogrid[:size[1], :size[0]]
        earth = (xx-cx)**2 + (yy-cy)**2 <= radius**2
        rows = np.flatnonzero(earth.any(axis=1))
        top, bottom = int(rows[0]), int(rows[-1])
        print(f'Source starts at {first / fps:.2f}s; duration {count / fps:.2f}s; Earth center {cx:.1f},{cy:.1f}, radius {radius:.1f}', flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        silent = OUTPUT_DIR / 'lightning_scan_silent.mp4'
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            raise RuntimeError('Could not create output video.')
        triggered_previous = np.zeros(earth.shape, dtype=np.uint8)
        previous_y = top
        previous_sweep = 0
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Could not decode frame {first + index}.')
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            sweep_progress = SCAN_PASSES * index / max(1, count-1)
            sweep = min(SCAN_PASSES - 1, int(sweep_progress))
            y = round(top + (bottom-top) * (sweep_progress - sweep))
            if sweep != previous_sweep:
                # Reset at the top; the jump is not a crossing of the whole disk.
                previous_y = y
                triggered_previous.fill(0)
                previous_sweep = sweep
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            yellow = cv2.inRange(hsv, (18, 110, 130), (40, 255, 255))
            yellow[~earth] = 0
            components, labels, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
            # Carry triggered dots across adjacent frames, allowing slight rotation.
            tracked_area = cv2.dilate(triggered_previous, np.ones((7, 7), np.uint8)) > 0
            carried = set(np.unique(labels[tracked_area]).tolist()) - {0}
            crossed = set(np.unique(labels[max(top, previous_y-1):min(bottom+1, y+2)]).tolist()) - {0}
            fresh = crossed - carried
            fresh = {label for label in fresh if stats[label, cv2.CC_STAT_AREA] >= 2}
            # Separate dots crossing simultaneously each produce a one-shot.
            events.extend([round(index * SAMPLE_RATE / fps)] * len(fresh))
            triggered_previous = np.isin(labels, list(carried | fresh)).astype(np.uint8) * 255
            previous_y = y
            line = np.zeros(earth.shape, dtype=np.uint8)
            cv2.line(line, (0, y), (size[0]-1, y), 255, 2)
            frame[(line > 0) & earth] = 255
            writer.write(frame)
            if index in {0, count//2, count-1}:
                cv2.imwrite(str(OUTPUT_DIR / f'scan_check_{index:04d}.png'), frame)
            if index % round(fps * 20) == 0:
                print(f'Rendered {index+1}/{count}; {len(events)} dot triggers', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    output = write_audio_video(sample, events, count / fps)
    import json
    (OUTPUT_DIR / 'lightning_events.json').write_text(json.dumps({
        'source_start_seconds': first/fps, 'duration_seconds': count/fps,
        'scan_passes': SCAN_PASSES, 'bell_fadeout_seconds': BELL_FADEOUT_SECONDS,
        'trigger_seconds': [s/SAMPLE_RATE for s in events]}, indent=2), encoding='utf-8')
    print(f'Saved {output}; {len(events)} one-shot events', flush=True)
    return output


def remix_existing():
    import json
    event_path = OUTPUT_DIR / 'lightning_events.json'
    metadata = json.loads(event_path.read_text(encoding='utf-8'))
    if metadata['source_start_seconds'] != START_SECONDS or metadata['scan_passes'] != SCAN_PASSES:
        raise ValueError('Cached events do not match the current scan settings; run render().')
    events = [round(t * SAMPLE_RATE) for t in metadata['trigger_seconds']]
    output = write_audio_video(combined_hit(), events, metadata['duration_seconds'])
    metadata.update(yellow_file=YELLOW_PATH.name, kick_file=KICK_PATH.name,
                    yellow_hit_seconds=YELLOW_HIT_SECONDS, bell_fadeout_seconds=BELL_FADEOUT_SECONDS,
                    yellow_tone_filter=YELLOW_TONE_FILTER)
    event_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Replaced audio using {len(events)} existing dot crossings: {output}', flush=True)
    return output


if __name__ == '__main__':
    render()
