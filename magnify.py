import cv2
import numpy as np
from metadata import MetaData
from scipy import fftpack as fftpack

class Magnify:
    def __init__(self, data: MetaData):
        self._data = data

    def load_video(self) -> (np.ndarray, int):
        cap = cv2.VideoCapture(self._in_file_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        x = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                video_tensor[x] = frame
                x += 1
            else:
                break
        return video_tensor, fps

    def save_video(self, video_tensor: np.ndarray) -> None:
        # four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        [height, width] = video_tensor[0].shape[0:2]
        writer = cv2.VideoWriter(self._out_file_name, four_cc, 30, (width, height), 1)
        for i in range(0, video_tensor.shape[0]):
            writer.write(cv2.convertScaleAbs(video_tensor[i]))
        writer.release()

    def do_magnify(self) -> None:
        tensor, fps = self.load_video()
        video_tensor = self._magnify_impl(tensor, fps)
        self.save_video(video_tensor)
        self._data.save_meta_data()

    def build_gaussian_pyramid(src, level=3):
        s = src.copy()
        pyramid = [s]
        for i in range(level):
            s = cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    def temporal_ideal_filter(tensor: np.ndarray, low: float, high: float, fps: int, axis: int = 0) -> np.ndarray:
        fft = fftpack.fft(tensor, axis=axis)
        frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff = fftpack.ifft(fft, axis=axis)
        return np.abs(iff)
    
    def gaussian_video(video_tensor, levels=3):
        for i in range(0, video_tensor.shape[0]):
            frame = video_tensor[i]
            pyr = build_gaussian_pyramid(frame, level=levels)
            gaussian_frame = pyr[-1]
            if i == 0:
                vid_data = np.zeros((video_tensor.shape[0], gaussian_frame.shape[0], gaussian_frame.shape[1], 3))
            vid_data[i] = gaussian_frame
        return vid_data

    def _magnify_impl(self, tensor: np.ndarray, fps: int) -> np.ndarray:
        gau_video = gaussian_video(tensor, levels=self._levels)
        filtered_tensor = temporal_ideal_filter(gau_video, self._low, self._high, fps)
        amplified_video = self._amplify_video(filtered_tensor)
        return self._reconstruct_video(amplified_video, tensor)

    def _amplify_video(self, gaussian_vid):
        return gaussian_vid * self._amplification

    def _reconstruct_video(self, amp_video, origin_video):
        origin_video_shape = origin_video.shape[1:]
        for i in range(0, amp_video.shape[0]):
            img = amp_video[i]
            for x in range(self._levels):
                img = cv2.pyrUp(img)  # this doubles the dimensions of img each time
            # ensure that dimensions are equal
            origin_video[i] += self._correct_dimensionality_problem_after_pyr_up(img, origin_video_shape)
        return origin_video

    def _correct_dimensionality_problem_after_pyr_up(self, img: np.ndarray, origin_video_frame_shape) -> np.ndarray:
        if img.shape != origin_video_frame_shape:
            return np.resize(img, origin_video_frame_shape)
        else:
            return img

    def principal_component_analysis(self, tensor: np.ndarray):
        # Data matrix tensor, assumes 0-centered
        n, m = tensor.shape
        assert np.allclose(tensor.mean(axis=0), np.zeros(m))
        # Compute covariance matrix
        covariance_matrix = np.dot(tensor.T, tensor) / (n - 1)
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)
        # Project tensor onto PC space
        X_pca = np.dot(tensor, eigen_vecs)
        return X_pca

    @property
    def _low(self) -> float:
        return self._data['low']

    @property
    def _high(self) -> float:
        return self._data['high']

    @property
    def _levels(self) -> int:
        return self._data['levels']

    @property
    def _amplification(self) -> float:
        return self._data['amplification']

    @property
    def _in_file_name(self) -> str:
        return self._data['file']

    @property
    def _out_file_name(self) -> str:
        return self._data['target']
