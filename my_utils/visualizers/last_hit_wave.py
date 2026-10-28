import cv2
import numpy as np


def create_wave(frame, center, frame_num, max_frames, max_radius):
    if max_frames <= 0 or max_radius <= 0 or frame_num < 0 or frame_num >= max_frames:
        return np.copy(frame)

    wave_frame = np.copy(frame)
    center_pt = (int(round(center[0])), int(round(center[1])))
    current_radius = int(max_radius * (frame_num / max_frames))

    for r in range(0, current_radius, 5):
        cv2.circle(wave_frame, center_pt, r, (0, 255, 255), -1)

    alpha = 1.0 - (frame_num / max_frames)
    cv2.addWeighted(wave_frame, alpha, frame, 1.0 - alpha, 0, wave_frame)
    return wave_frame


class Wave:
    def __init__(self, position, start_frame, max_frames=5, max_radius=15):
        if max_frames <= 0 or max_radius <= 0:
            raise ValueError("max_frames and max_radius must be positive")
        self.position = (int(round(position[0])), int(round(position[1])))
        self.start_frame = int(start_frame)
        self.max_frames = int(max_frames)
        self.max_radius = int(max_radius)
        self.active = True

    def update(self, frame, current_frame):
        frame_num = int(current_frame) - self.start_frame
        if 0 <= frame_num < self.max_frames:
            return create_wave(frame, self.position, frame_num, self.max_frames, self.max_radius)
        self.active = False
        return frame
