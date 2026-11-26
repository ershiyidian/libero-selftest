from pathlib import Path

import imageio
import numpy as np


class LiberoDualViewRecorder:
    def __init__(self, path: Path, fps: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.path),
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            quality=6,
            macro_block_size=16,
        )
        self.frames = 0

    def append(self, observation):
        front = np.asarray(observation["agentview_image"])
        wrist = np.asarray(observation["robot0_eye_in_hand_image"])
        if front.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
            raise ValueError("The real LIBERO RGB camera images must be 256x256x3")
        if front.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("The LIBERO RGB camera images must be uint8")
        image = np.concatenate((front[::-1, ::-1], wrist[::-1, ::-1]), axis=1)
        self.writer.append_data(np.ascontiguousarray(image))
        self.frames += 1

    def close(self):
        self.writer.close()
