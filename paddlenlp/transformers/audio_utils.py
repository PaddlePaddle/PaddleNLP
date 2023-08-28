# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Audio processing functions to extract features from audio waveforms. This code is pure numpy to support all frameworks
and remove unnecessary dependencies.
"""
import warnings
from typing import Optional, Union

import numpy as np


def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from hertz to mels.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies on the mel scale.
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from mels to hertz.

    Args:
        mels (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in mels.
        mel_scale (`str`, *optional*, `"htk"`):
            The mel frequency scale to use, `"htk"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies in hertz.
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.

    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins in Hz.
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            Center frequencies of the triangular filters to create, in Hz.

    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.ndarray:
    """
    Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
    various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
    are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

    Different banks of mel filters were introduced in the literature. The following variations are supported:

    - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
      bandwidth of `[0, 4600]` Hz.
    - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
      bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
    - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
      speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
      12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

    This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
    `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

    Args:
        num_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        num_mel_filters (`int`):
            Number of mel filters to generate.
        min_frequency (`float`):
            Lowest frequency of interest in Hz.
        max_frequency (`float`):
            Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
        sampling_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"` or `"slaney"`.

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
        projection matrix to go from a spectrogram to a mel spectrogram.
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # frequencies of FFT bins in Hz
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters


def optimal_fft_length(window_length: int) -> int:
    """
    Finds the best FFT input size for a given `window_length`. This function takes a given window length and, if not
    already a power of two, rounds it up to the next power or two.

    The FFT algorithm works fastest when the length of the input is a power of two, which may be larger than the size
    of the window or analysis frame. For example, if the window is 400 samples, using an FFT input size of 512 samples
    is more optimal than an FFT size of 400 samples. Using a larger FFT size does not affect the detected frequencies,
    it simply gives a higher frequency resolution (i.e. the frequency bins are smaller).
    """
    return 2 ** int(np.ceil(np.log2(window_length)))


def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the specified window. This window is intended to be used with `stft`.

    The following window types are supported:

        - `"boxcar"`: a rectangular window
        - `"hamming"`: the Hamming window
        - `"hann"`: the Hann window

    Args:
        window_length (`int`):
            The length of the window in samples.
        name (`str`, *optional*, defaults to `"hann"`):
            The name of the window function.
        periodic (`bool`, *optional*, defaults to `True`):
            Whether the window is periodic or symmetric.
        frame_length (`int`, *optional*):
            The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
            than the frame length, so that it will be zero-padded.
        center (`bool`, *optional*, defaults to `True`):
            Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

    Returns:
        `np.ndarray` of shape `(window_length,)` or `(frame_length,)` containing the window.
    """
    length = window_length + 1 if periodic else window_length

    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    else:
        raise ValueError(f"Unknown window function '{name}'")

    if periodic:
        window = window[:-1]

    if frame_length is None:
        return window

    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window


# TODO This method does not support batching yet as we are mainly focused on inference.
def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.

    This function can create the following kinds of spectrograms:

      - amplitude spectrogram (`power = 1.0`)
      - power spectrogram (`power = 2.0`)
      - complex-valued spectrogram (`power = None`)
      - log spectrogram (use `log_mel` argument)
      - mel spectrogram (provide `mel_filters`)
      - log-mel spectrogram (provide `mel_filters` and `log_mel`)

    How this works:

      1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `frame_length
         - hop_length` samples.
      2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
      3. The DFT is taken of each windowed frame.
      4. The results are stacked into a spectrogram.

    We make a distinction between the following "blocks" of sample data, each of which may have a different lengths:

      - The analysis frame. This is the size of the time slices that the input waveform is split into.
      - The window. Each analysis frame is multiplied by the window to avoid spectral leakage.
      - The FFT input buffer. The length of this determines how many frequency bins are in the spectrogram.

    In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame. A
    padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
    typically the next power of two.

    Note: This function is not optimized for speed yet. It should be mostly compatible with `librosa.stft` and
    `torchaudio.functional.transforms.Spectrogram`, although it is more flexible due to the different ways spectrograms
    can be constructed.

    Args:
        waveform (`np.ndarray` of shape `(length,)`):
            The input waveform. This must be a single real-valued, mono waveform.
        window (`np.ndarray` of shape `(frame_length,)`):
            The windowing function to apply, including zero-padding if necessary. The actual window length may be
            shorter than `frame_length`, but we're assuming the array has already been zero-padded.
        frame_length (`int`):
            The length of the analysis frames in samples. With librosa this is always equal to `fft_length` but we also
            allow smaller sizes.
        hop_length (`int`):
            The stride between successive analysis frames in samples.
        fft_length (`int`, *optional*):
            The size of the FFT buffer in samples. This determines how many frequency bins the spectrogram will have.
            For optimal speed, this should be a power of two. If `None`, uses `frame_length`.
        power (`float`, *optional*, defaults to 1.0):
            If 1.0, returns the amplitude spectrogram. If 2.0, returns the power spectrogram. If `None`, returns
            complex numbers.
        center (`bool`, *optional*, defaults to `True`):
            Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
            `t` will start at time `t * hop_length`.
        pad_mode (`str`, *optional*, defaults to `"reflect"`):
            Padding mode used when `center` is `True`. Possible values are: `"constant"` (pad with zeros), `"edge"`
            (pad with edge values), `"reflect"` (pads with mirrored values).
        onesided (`bool`, *optional*, defaults to `True`):
            If True, only computes the positive frequencies and returns a spectrogram containing `fft_length // 2 + 1`
            frequency bins. If False, also computes the negative frequencies and returns `fft_length` frequency bins.
        preemphasis (`float`, *optional*)
            Coefficient for a low-pass filter that applies pre-emphasis before the DFT.
        mel_filters (`np.ndarray` of shape `(num_freq_bins, num_mel_filters)`, *optional*):
            The mel filter bank. If supplied, applies a this filter bank to create a mel spectrogram.
        mel_floor (`float`, *optional*, defaults to 1e-10):
            Minimum value of mel frequency banks.
        log_mel (`str`, *optional*):
            How to convert the spectrogram to log scale. Possible options are: `None` (don't convert), `"log"` (take
            the natural logarithm) `"log10"` (take the base-10 logarithm), `"dB"` (convert to decibels). Can only be
            used when `power` is not `None`.
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-10`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. For a power spectrogram, the default of `1e-10` corresponds to a minimum of -100 dB. For an
            amplitude spectrogram, the value `1e-5` corresponds to -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            Data type of the spectrogram tensor. If `power` is None, this argument is ignored and the dtype will be
            `np.complex64`.

    Returns:
        `nd.array` containing a spectrogram of shape `(num_frequency_bins, length)` for a regular spectrogram or shape
        `(num_mel_filters, length)` for a mel spectrogram.
    """
    window_length = len(window)

    if fft_length is None:
        fft_length = frame_length

    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")

    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    # center pad the waveform
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)

    # promote to float64, since np.fft uses float64 internally
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)

    # split waveform into frames of frame_length size
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # rfft is faster than fft
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # note: ** is much faster than np.power
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

    spectrogram = spectrogram.T

    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        spectrogram = np.asarray(spectrogram, dtype)

    return spectrogram


def power_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:
    """
    Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`, using basic
    logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
    This means that large variations in energy may not sound all that different if the sound is loud to begin with.
    This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

    Based on the implementation of `librosa.power_to_db`.

    Args:
        spectrogram (`np.ndarray`):
            The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-10`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

    Returns:
        `np.ndarray`: the spectrogram in decibels
    """
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram


def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
    """
    Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
    basic logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
    This means that large variations in energy may not sound all that different if the sound is loud to begin with.
    This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

    Args:
        spectrogram (`np.ndarray`):
            The input amplitude (mel) spectrogram.
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-5`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

    Returns:
        `np.ndarray`: the spectrogram in decibels
    """
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram


def get_mel_filter_banks(
    nb_frequency_bins: int,
    nb_mel_filters: int,
    frequency_min: float,
    frequency_max: float,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.array:
    warnings.warn(
        "The function `get_mel_filter_banks` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    return mel_filter_bank(
        num_frequency_bins=nb_frequency_bins,
        num_mel_filters=nb_mel_filters,
        min_frequency=frequency_min,
        max_frequency=frequency_max,
        sampling_rate=sample_rate,
        norm=norm,
        mel_scale=mel_scale,
    )


def fram_wave(waveform: np.array, hop_length: int = 160, fft_window_size: int = 400, center: bool = True):
    """
    In order to compute the short time fourier transform, the waveform needs to be split in overlapping windowed
    segments called `frames`.

    The window length (window_length) defines how much of the signal is contained in each frame, while the hop length
    defines the step between the beginning of each new frame.


    Args:
        waveform (`np.array` of shape `(sample_length,)`):
            The raw waveform which will be split into smaller chunks.
        hop_length (`int`, *optional*, defaults to 160):
            Step between each window of the waveform.
        fft_window_size (`int`, *optional*, defaults to 400):
            Defines the size of the window.
        center (`bool`, defaults to `True`):
            Whether or not to center each frame around the middle of the frame. Centering is done by reflecting the
            waveform on the left and on the right.

    Return:
        framed_waveform (`np.array` of shape `(waveform.shape // hop_length , fft_window_size)`):
            The framed waveforms that can be fed to `np.fft`.
    """
    warnings.warn(
        "The function `fram_wave` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    frames = []
    for i in range(0, waveform.shape[0] + 1, hop_length):
        if center:
            half_window = (fft_window_size - 1) // 2 + 1
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]
            frame = waveform[start:end]
            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            elif end == waveform.shape[0]:
                padd_width = (0, (i - waveform.shape[0] + half_window))
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")

        else:
            frame = waveform[i : i + fft_window_size]
            frame_width = frame.shape[0]
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(
                    frame, pad_width=(0, fft_window_size - frame_width), mode="constant", constant_values=0
                )
        frames.append(frame)

    frames = np.stack(frames, 0)
    return frames


def stft(frames: np.array, windowing_function: np.array, fft_window_size: int = None):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same results
    as `torch.stft`.

    Args:
        frames (`np.array` of dimension `(num_frames, fft_window_size)`):
            A framed audio signal obtained using `audio_utils.fram_wav`.
        windowing_function (`np.array` of dimension `(nb_frequency_bins, nb_mel_filters)`:
            A array reprensenting the function that will be used to reduces the amplitude of the discontinuities at the
            boundaries of each frame when computing the STFT. Each frame will be multiplied by the windowing_function.
            For more information on the discontinuities, called *Spectral leakage*, refer to [this
            tutorial]https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
        fft_window_size (`int`, *optional*):
            Size of the window om which the Fourier transform is applied. This controls the frequency resolution of the
            spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples. The number of
            frequency bins (`nb_frequency_bins`) used to divide the window into equal strips is equal to
            `(1+fft_window_size)//2`. An increase of the fft_window_size slows the calculus time proportionnally.

    Example:

    ```python
    >>> from paddlenlp.transformers.audio_utils import stft, fram_wave
    >>> import numpy as np

    >>> audio = np.random.rand(50)
    >>> fft_window_size = 10
    >>> hop_length = 2
    >>> framed_audio = fram_wave(audio, hop_length, fft_window_size)
    >>> spectrogram = stft(framed_audio, np.hanning(fft_window_size + 1))
    ```

    Returns:
        spectrogram (`np.ndarray`):
            A spectrogram of shape `(num_frames, nb_frequency_bins)` obtained using the STFT algorithm
    """
    warnings.warn(
        "The function `stft` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    frame_size = frames.shape[1]

    if fft_window_size is None:
        fft_window_size = frame_size

    if fft_window_size < frame_size:
        raise ValueError("FFT size must greater or equal the frame size")
    # number of FFT bins to store
    nb_frequency_bins = (fft_window_size >> 1) + 1

    spectrogram = np.empty((len(frames), nb_frequency_bins), dtype=np.complex64)
    fft_signal = np.zeros(fft_window_size)

    for f, frame in enumerate(frames):
        if windowing_function is not None:
            np.multiply(frame, windowing_function, out=fft_signal[:frame_size])
        else:
            fft_signal[:frame_size] = frame
        spectrogram[f] = np.fft.fft(fft_signal, axis=0)[:nb_frequency_bins]
    return spectrogram.T
