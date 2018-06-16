import collections
import contextlib
import sys
import wave

import webrtcvad

import librosa
import librosa.display
import numpy as np
import mfccProcess as mf

from pydub import AudioSegment
AudioSegment.converter = "ffmpeg/fmpeg.exe"


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def main(args):
    audio, sample_rate = read_wave('final_data/user.wav')
    vad = webrtcvad.Vad(3)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 10, 300, vad, frames)

    for i, segment in enumerate(segments):
        if i==2:
            sys.stderr.write('공백 없이 다시 녹음해주시기 바랍니다.')
            sys.exit(1)
        path = 'user_vad.wav'
        print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)

    sound = AudioSegment.from_file(path, "wav")
    normalized_sound = match_target_amplitude(sound, -20.0)
    normalized_sound.export('user_final.wav', format="wav")

    good, sra = librosa.load('galak.wav')
    zcr_good = []
    zcr_good = librosa.feature.zero_crossing_rate(good)
    len_good = len(zcr_good[0])

    user, srb = librosa.load('user_final.wav')
    zcr_user = []
    zcr_user = librosa.feature.zero_crossing_rate(user)
    len_user = len(zcr_user[0])

    if len_user < len_good:
        lenf = len_user
    else:
        lenf = len_good

    tot = 0
    for i in range(0, lenf):
        tot += abs(zcr_user[0, i]-zcr_good[0, i])/zcr_good[0, i] *100
    print("결과: ")
    zcr_res = 100-(tot/lenf)
    print(zcr_res)

    pit_good, mag_good = librosa.core.piptrack(y=good, sr=sra)
    pit_user, mag_user = librosa.core.piptrack(y=user, sr=srb)

    tot = 0
    cnt_user = 0
    cnt_good = 0

    index_good = mag_good[:, 0].argmax()
    pre_res_good = pit_good[index_good, 0]

    index_user = mag_user[:, 0].argmax()
    pre_res_user = pit_user[index_user, 0]

    print("길이",len(mag_good[0]),len(mag_user[0]))

    for t in range(1, len(mag_good[0])):
        index_good = mag_good[:, t].argmax()
        res_good = pit_good[index_good, t]
        if abs(pre_res_good-res_good) > 300:
            cnt_good += 1
        pre_res_good = res_good

    for t in range(1, len(mag_user[0])):
        index_user = mag_user[:, t].argmax()
        res_user = pit_user[index_user, t]

        if abs(pre_res_user-res_user) > 300:
            cnt_user += 1
        pre_res_user = res_user

    print("결과: ")
    print(cnt_user,cnt_good)
    print(cnt_user-cnt_good)
    fre_res = 100-(abs(cnt_good-cnt_user) / cnt_good)*100
    print(fre_res)

    mfcc_res = mf.mfcc_def(good, sra, user, srb)

    print("완전 결과")
    print((zcr_res+fre_res+mfcc_res)/3)


if __name__ == '__main__':
    main(sys.argv[1:])
