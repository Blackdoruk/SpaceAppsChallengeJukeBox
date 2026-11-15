from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pixasonics.core import Mapper
from pixasonics.features import Feature


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "GHGRightPanel.mp4"
AUDIO_TRACKS: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    "Yellow": ("yellow.mp3", ((24, 36),)),
    "Orange": ("orange.mp3", ((8, 23),)),
    "Red": ("red.mp3", ((0, 7), (173, 179))),
}
BACKGROUND_AUDIO_PATH = SCRIPT_DIR.parent / "AirQuality" / "co_final.mp3"
OUTPUT_PATH = SCRIPT_DIR / "greenhouse_gas_soundification.mp4"

OUTPUT_WIDTH = 1280
SWEEP_ROUND_TRIPS = 3
SCAN_EXTENT = 0.82
SCAN_BAND_HALF_WIDTH = 6
MIN_SATURATION = 65
MIN_VALUE = 35
TRACK_GAIN_MAX = 0.85
TRACK_GAIN_EXPONENT = 1.5
GAIN_ATTACK_SECONDS = 0.35
GAIN_RELEASE_SECONDS = 1.2
BACKGROUND_GAIN = 0.12
SAMPLE_RATE = 44100
SAMPLE_RMS_TARGET_DBFS = -20.0
SAMPLE_PEAK_CEILING_DBFS = -2.0
LOOP_CROSSFADE_SECONDS = 0.15
VIDEO_AUDIO_FADE_SECONDS = 0.5
TRAILING_SILENCE_THRESHOLD_DBFS = -50.0


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def count_decodable_frames(ffprobe: str, video_path: Path) -> int:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "quiet",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        frame_count = int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            f"Could not determine decodable frame count for {video_path}."
        ) from error
    if frame_count <= 0:
        raise ValueError(f"No decodable video frames found in {video_path}.")
    return frame_count


def scan_column(
    frame_index: int,
    frame_count: int,
    center_x: int,
    radius: int,
) -> int:
    progress = frame_index / max(frame_count - 1, 1)
    sweep = 0.5 - 0.5 * math.cos(
        2 * math.pi * SWEEP_ROUND_TRIPS * progress
    )
    return round(center_x + (2 * sweep - 1) * SCAN_EXTENT * radius)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def scanline_color_strength(
    hsv: np.ndarray,
    x: int,
    center_x: int,
    center_y: int,
    radius: int,
    hue_ranges: tuple[tuple[int, int], ...],
) -> float:
    height, width = hsv.shape[:2]
    left = max(0, x - SCAN_BAND_HALF_WIDTH)
    right = min(width, x + SCAN_BAND_HALF_WIDTH + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)

    y_coords, x_coords = np.ogrid[top:bottom, left:right]
    inside_sphere = (
        (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        <= radius**2
    )
    pixels_in_sphere = int(np.count_nonzero(inside_sphere))
    if pixels_in_sphere == 0:
        return 0.0

    scanline = hsv[top:bottom, left:right]
    hue = scanline[:, :, 0]
    color_mask = np.zeros(hue.shape, dtype=bool)
    for hue_low, hue_high in hue_ranges:
        color_mask |= (hue >= hue_low) & (hue <= hue_high)
    color_mask &= (scanline[:, :, 1] >= MIN_SATURATION) & (
        scanline[:, :, 2] >= MIN_VALUE
    )
    selected = color_mask & inside_sphere
    strength = (
        scanline[:, :, 1].astype(np.float32) / 255.0
    ) * (scanline[:, :, 2].astype(np.float32) / 255.0)
    return float(strength[selected].sum() / pixels_in_sphere)


def smooth_values(values: np.ndarray, fps: float) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=np.float32)
    previous = 0.0
    for index, target in enumerate(values):
        duration = (
            GAIN_ATTACK_SECONDS
            if target > previous
            else GAIN_RELEASE_SECONDS
        )
        alpha = 1.0 - math.exp(-1.0 / (fps * duration))
        previous += alpha * (float(target) - previous)
        smoothed[index] = previous
    return smoothed


def decode_mp3(mp3_path: Path, ffmpeg: str) -> np.ndarray:
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(mp3_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0 or audio.size % 2:
        raise ValueError(f"Decoded MP3 is empty or malformed: {mp3_path}")
    audio = audio.reshape(-1, 2)
    if not np.isfinite(audio).all():
        raise ValueError(f"MP3 contains non-finite audio samples: {mp3_path}")

    block_size = round(SAMPLE_RATE * 0.1)
    silence_threshold = 10 ** (TRAILING_SILENCE_THRESHOLD_DBFS / 20)
    last_active_sample = 0
    for start in range(0, len(audio), block_size):
        block = audio[start : start + block_size]
        block_rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        if block_rms >= silence_threshold:
            last_active_sample = start + len(block)
    if last_active_sample == 0:
        raise ValueError(f"MP3 is silent: {mp3_path}")
    tail_samples = round(SAMPLE_RATE * 0.1)
    audio = audio[: min(len(audio), last_active_sample + tail_samples)]

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms <= 1e-8 or peak <= 1e-8:
        raise ValueError(f"MP3 is silent: {mp3_path}")
    target_rms = 10 ** (SAMPLE_RMS_TARGET_DBFS / 20)
    peak_ceiling = 10 ** (SAMPLE_PEAK_CEILING_DBFS / 20)
    gain = min(target_rms / rms, peak_ceiling / peak)
    print(f"{mp3_path.name} loop segment: {len(audio) / SAMPLE_RATE:.2f}s")
    return (audio * gain).astype(np.float32)


def loop_with_crossfade(
    audio: np.ndarray,
    target_samples: int,
) -> np.ndarray:
    crossfade = min(
        round(SAMPLE_RATE * LOOP_CROSSFADE_SECONDS),
        len(audio) // 4,
    )
    if crossfade < 2:
        raise ValueError("An MP3 is too short for a seamless loop.")

    output = audio.copy()
    while len(output) < target_samples:
        fade = np.linspace(
            0.0,
            1.0,
            crossfade,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
        overlap = output[-crossfade:] * (1.0 - fade)
        overlap += audio[:crossfade] * fade
        output = np.concatenate(
            (output[:-crossfade], overlap, audio[crossfade:]),
            axis=0,
        )
    return output[:target_samples]


def write_stereo_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    require_file(VIDEO_PATH)
    audio_paths = {
        track_name: SCRIPT_DIR / filename
        for track_name, (filename, _) in AUDIO_TRACKS.items()
    }
    for audio_path in audio_paths.values():
        require_file(audio_path)
    require_file(BACKGROUND_AUDIO_PATH)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError("FFmpeg and ffprobe must be available on PATH.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (
        fps <= 0
        or expected_frames <= 0
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError("Source video has invalid frame metadata.")
    decodable_frames = count_decodable_frames(ffprobe, VIDEO_PATH)
    if decodable_frames != expected_frames:
        print(
            "Warning: source metadata reports "
            f"{expected_frames} frames, but only {decodable_frames} are decodable."
        )
    expected_frames = decodable_frames

    output_width = OUTPUT_WIDTH
    output_height = OUTPUT_WIDTH
    center_x = output_width // 2
    center_y = output_height // 2
    radius = round(min(output_width, output_height) * 0.43)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    scores_by_track = {track_name: [] for track_name in AUDIO_TRACKS}
    columns = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index = len(columns)
        reduced = resize_frame(frame, output_width, output_height)
        x = scan_column(frame_index, expected_frames, center_x, radius)
        hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
        for track_name, (_, hue_ranges) in AUDIO_TRACKS.items():
            scores_by_track[track_name].append(
                scanline_color_strength(
                    hsv,
                    x,
                    center_x,
                    center_y,
                    radius,
                    hue_ranges,
                )
            )
        columns.append(x)
        if frame_index and frame_index % 600 == 0:
            print(f"Analyzed {frame_index}/{expected_frames} frames")
    cap.release()

    if not columns:
        raise RuntimeError("No frames could be decoded from the source video.")
    if len(columns) != expected_frames:
        print(
            f"Warning: OpenCV decoded {len(columns)}/{expected_frames} frames; "
            "the output will use only decodable frames."
        )
    score_arrays = {
        track_name: np.asarray(scores, dtype=np.float32)
        for track_name, scores in scores_by_track.items()
    }
    gain_arrays = {}
    for track_name, scores in score_arrays.items():
        positive_scores = scores[scores > 0]
        if positive_scores.size == 0:
            raise ValueError(
                f"No {track_name} color was detected under the vertical line. "
                "Check the configured HSV range."
            )
        intensity_high = float(np.percentile(positive_scores, 98))
        feature = Feature(name=f"{track_name} color density")
        mapper = Mapper(
            source=feature,
            target={
                "owner": SimpleNamespace(num_channels=1),
                "name": f"{track_name} MP3 volume",
                "param_name": "amplitude",
            },
            in_low=0.0,
            in_high=intensity_high,
            out_low=0.0,
            out_high=TRACK_GAIN_MAX,
            exponent=TRACK_GAIN_EXPONENT,
            name=f"{track_name} density to MP3 volume",
        )
        target_gains = np.empty_like(scores)
        for index, intensity in enumerate(scores):
            feature.features = np.array([[intensity]], dtype=np.float64)
            mapped = mapper.map(feature.features.data)
            target_gains[index] = mapper.clamp_mappings(mapped)[0][0, 0]
        gain_arrays[track_name] = smooth_values(target_gains, fps)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_audio_samples = round(len(columns) / fps * SAMPLE_RATE)
    sample_positions = (
        np.arange(target_audio_samples, dtype=np.float64) * fps / SAMPLE_RATE
    )
    mixed_audio = np.zeros((target_audio_samples, 2), dtype=np.float32)
    for track_name, audio_path in audio_paths.items():
        track_audio = decode_mp3(audio_path, ffmpeg)
        track_audio = loop_with_crossfade(track_audio, target_audio_samples)
        audio_envelope = np.interp(
            sample_positions,
            np.arange(len(gain_arrays[track_name]), dtype=np.float64),
            gain_arrays[track_name],
        ).astype(np.float32)
        mixed_audio += track_audio * audio_envelope[:, None]

    background_audio = decode_mp3(BACKGROUND_AUDIO_PATH, ffmpeg)
    background_audio = loop_with_crossfade(
        background_audio,
        target_audio_samples,
    )
    mixed_audio += background_audio * BACKGROUND_GAIN

    fade_samples = min(
        round(SAMPLE_RATE * VIDEO_AUDIO_FADE_SECONDS),
        target_audio_samples // 2,
    )
    if fade_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        mixed_audio[:fade_samples] *= fade_in[:, None]
        mixed_audio[-fade_samples:] *= fade_out[:, None]
    mixed_peak = float(np.max(np.abs(mixed_audio)))
    if mixed_peak > 0.95:
        mixed_audio *= 0.95 / mixed_peak

    with tempfile.TemporaryDirectory(prefix="greenhouse-gas-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        silent_video = temp_dir_path / "vertical_scan_silent.mp4"
        audio_wav = temp_dir_path / "greenhouse_audio.wav"
        write_stereo_wav(audio_wav, mixed_audio)

        writer = cv2.VideoWriter(
            str(silent_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the temporary scan video.")

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        for frame_index in range(len(columns)):
            ok, frame = cap.read()
            if not ok:
                writer.release()
                cap.release()
                raise RuntimeError(
                    f"Source video ended unexpectedly at frame {frame_index}."
                )
            reduced = resize_frame(frame, output_width, output_height)
            x = columns[frame_index]
            distance_x = x - center_x
            half_height = int(
                math.sqrt(max(radius * radius - distance_x * distance_x, 0))
            )
            y0 = max(0, center_y - half_height)
            y1 = min(output_height - 1, center_y + half_height)
            overlay = reduced.copy()
            cv2.line(
                overlay,
                (x, y0),
                (x, y1),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            reduced = cv2.addWeighted(overlay, 0.75, reduced, 0.25, 0)
            writer.write(reduced)
            if frame_index and frame_index % 600 == 0:
                print(f"Rendered {frame_index}/{len(columns)} frames")
        cap.release()
        writer.release()

        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(audio_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(OUTPUT_PATH),
            ],
            check=True,
        )

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height",
            "-of",
            "default=noprint_wrappers=1",
            str(OUTPUT_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Frames analyzed: {len(columns)} at {fps:.3f} fps")
    for track_name in AUDIO_TRACKS:
        scores = score_arrays[track_name]
        gains = gain_arrays[track_name]
        active_frames = int(np.count_nonzero(scores))
        print(
            f"{track_name} density: {scores.min():.6f}–{scores.max():.6f}; "
            f"detected in {active_frames}/{len(scores)} frames; "
            f"gain: {gains.min():.3f}–{gains.max():.3f}"
        )
    print(f"Created: {OUTPUT_PATH}")
    print(probe.stdout.strip())


if __name__ == "__main__":
    main()
