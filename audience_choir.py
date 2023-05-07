import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import wave
from collections import deque
import time
import librosa
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# カメラの設定
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 10
cap.set(cv2.CAP_PROP_FPS, fps)


# 音程を扱う設定
pitch = [
    261.626,
    277.183,
    293.665,
    311.127,
    329.628,
    349.228,
    369.994,
    391.995,
    415.305,
    440,
    466.164,
    493.883,
]
dia_to_chroma = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}
tone = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

root_idx = -1

pitch_idx = [root_idx + 0, root_idx + 4, root_idx + 7]
dia_idx = [0, 2, 4]


# 初期設定用の変数
rec_st = 0
proc_end = 0


# 左右の指の横の位置を保持する変数
l_c_r_l = 0
l_c_r_r = 0

updown = [0, 0, 0]  # [l, c, r] 1, -1, 0
lcr = ["l", "c", "r"]
is_gu = [0, 0, 0]

is_gu_l = 0
is_gu_r = 0

cap_end = 0


# 指の位置に四角を書く
def draw_rec_l(image):
    global l_c_r_l
    global width
    global height

    if l_c_r_l == "l":
        cv2.rectangle(image, (0, 0), (int(width / 3), height), (255, 0, 0), 10)
    elif l_c_r_l == "c":
        cv2.rectangle(
            image, (int(width / 3), 0), (int(width * 2 / 3), height), (255, 0, 0), 10
        )
    else:
        cv2.rectangle(image, (int(width * 2 / 3), 0), (width, height), (255, 0, 0), 10)


def draw_rec_r(image):
    global l_c_r_r
    global width
    global height

    if l_c_r_r == "l":
        cv2.rectangle(image, (0, 0), (int(width / 3), height), (0, 255, 0), 10)
    elif l_c_r_r == "c":
        cv2.rectangle(
            image, (int(width / 3), 0), (int(width * 2 / 3), height), (0, 255, 0), 10
        )
    else:
        cv2.rectangle(image, (int(width * 2 / 3), 0), (width, height), (0, 255, 0), 10)


# 指が左右のどの領域にいるか判定する
def check_l_c_r_l(index_tip_l):
    global l_c_r_l

    if index_tip_l[0] < 1 / 3:
        l_c_r_l = "l"
    elif index_tip_l[0] < 2 / 3:
        l_c_r_l = "c"
    else:
        l_c_r_l = "r"


def check_l_c_r_r(index_tip_r):
    global l_c_r_r

    if index_tip_r[0] < 1 / 3:
        l_c_r_r = "l"
    elif index_tip_r[0] < 2 / 3:
        l_c_r_r = "c"
    else:
        l_c_r_r = "r"


# 音について扱う
def man_audio():
    # global sample

    global proc_end
    global tone
    global pitch
    global rec_st
    global root_idx

    global pitch_idx
    global is_gu

    global cap_end

    RECORD_SECONDS = 3
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 2**10
    while 1:
        if rec_st:
            break
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []
    for i in range(int(RECORD_SECONDS * RATE / CHUNK)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("end")

    waveFile = wave.open("./output/audio4preset.wav", "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(pa.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b"".join(frames))
    waveFile.close()

    time.sleep(3)
    y, sr = librosa.load("./output/audio4preset.wav", mono=True)
    # print(sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    # times = librosa.times_like(f0)

    # 欠損値と外れ値を除いて平均をとる
    df = pd.Series(f0)
    df = df.dropna()

    def find_outliers(input_array):
        q1, q3 = np.percentile(input_array, [25, 75])
        iqr = q3 - q1

        return input_array[
            (q1 - 1.5 * iqr <= input_array) & (input_array <= q3 + 1.5 * iqr)
        ]

    df = find_outliers(df)
    print(df.mean())
    f0 = df.mean()

    octave_range = [65.406, 130.813, 261.626, 523.251, 1046.502]
    octave = 0

    # オクターブを定める
    def check_octave(f0):
        for i in range(len(octave_range) - 1):
            if octave_range[i] <= f0 < octave_range[i + 1]:
                return i - 2
        raise ValueError("frequency out of range")

    octave = check_octave(f0)

    # tone = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    # pitch = [261.626, 277.183, 293.665, 311.127, 329.628, 349.228, 369.994, 391.995, 415.305, 440, 466.164, 493.883]

    pitch = [i * (2**octave) for i in pitch]
    pitch_log_diff = [abs(np.log(i) - np.log(f0)) for i in pitch]

    # ルートを決定
    root_idx = pitch_log_diff.index(min(pitch_log_diff))
    root = pitch[root_idx]

    root_name = tone[root_idx]
    print(root_name, root)
    proc_end = 1

    ###############################

    pitch_idx = [root_idx + 0, root_idx + 4, root_idx + 7]

    per_length = 0.5

    sample = [
        np.sin(
            np.arange(44100 * per_length)
            * pitch[pitch_idx[0] % 12]
            * 2 ** int(pitch_idx[0] // 12)
            * np.pi
            * 2
            / 44100
        ),
        np.sin(
            np.arange(44100 * per_length)
            * pitch[pitch_idx[1] % 12]
            * 2 ** int(pitch_idx[1] // 12)
            * np.pi
            * 2
            / 44100
        ),
        np.sin(
            np.arange(44100 * per_length)
            * pitch[pitch_idx[2] % 12]
            * 2 ** int(pitch_idx[2] // 12)
            * np.pi
            * 2
            / 44100
        ),
    ]

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=44100,
        frames_per_buffer=1024,
        output=True,
    )

    # 位相をそろえるための変数
    last = [0, 0, 0]
    dy = [0, 0, 0]

    while 1:
        for i in range(3):
            if dy[i] >= 0:
                sample[i] = np.sin(
                    np.arange(44100 * per_length)
                    * pitch[pitch_idx[i] % 12]
                    * 2 ** int(pitch_idx[i] // 12)
                    * np.pi
                    * 2
                    / 44100
                    + last[i]
                )
            else:
                sample[i] = np.sin(
                    np.arange(44100 * per_length)
                    * pitch[pitch_idx[i] % 12]
                    * 2 ** int(pitch_idx[i] // 12)
                    * np.pi
                    * 2
                    / 44100
                    + np.pi
                    - last[i]
                )

            last[i] = np.arcsin(sample[i][-1])
            dy[i] = sample[i][-1] - sample[i][-2]

            if is_gu[i]:
                sample[i] = np.arange(44100 * per_length) * 0

        stream.write(
            ((sample[0] + sample[1] + sample[2]) / 3).astype(np.float32).tostring()
        )

        if cap_end:
            stream.close()
            break
        # if keyboard.read_key() == "escape":
        #    break


# 変位の情報をためるバッファをリセット
def reset_buff(buff, buff_size):
    for i in range(buff_size):
        buff.popleft()
        buff.append(0)


# 上下運動を判断
def check_buff(buff):
    up = 0
    down = 0
    if buff[0] > 0:
        up = 1
        for i in range(1, len(buff)):
            if buff[i] <= 0:
                up = 0
        if up:
            return 1
    elif buff[0] < 0:
        down = 1
        for i in range(1, len(buff)):
            if buff[i] >= 0:
                down = 0
        if down:
            return -1

    return 0


"""
def set_pitch():
    global root_idx
    global pitch_idx
    global dia_idx
    global dia_to_chroma

    while(1):
        for i in range(3):
            pitch_idx[i] = root_idx + int(dia_idx[i]//7)*12 + dia_to_chroma[dia_idx[i]%7]
            #pitch[i] = 12*(pitch_idx[i]//7) + pitch_to_idx[pitch_idx[i]%7]
"""

# グーになっているかチェック
loaded_model = pickle.load(open("is_gu.sav", "rb"))


def check_is_gu(landmark):
    global loaded_model
    # data = results.multi_hand_landmarks[0].landmark
    data = landmark
    df = []
    for i in range(21):
        df.append(data[i].x - data[0].x)
        df.append(data[i].y - data[0].y)
        df.append(data[i].z - data[0].z)
    df = pd.DataFrame(df).T
    # is_gu = knn.predict(df)
    is_gu = loaded_model.predict(df)

    return is_gu


def showVideo():
    global cap
    global fps
    global width
    global height
    global l_c_r_l
    global l_c_r_r
    global updown
    global pitch_idx

    global dia_idx

    global rec_st
    global proc_end

    global is_gu
    global is_gu_l
    global is_gu_r

    global cap_end

    cv2.namedWindow("display", cv2.WINDOW_NORMAL)

    preset_count = 3 * fps - 1

    # カウントダウン + recording中の画面
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)

        if preset_count > 0:
            cv2.putText(
                image,
                text=str(int(preset_count // fps + 1)),
                org=(int(width // 2), int(height // 2)),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=2.0,
                color=(0, 0, 255),
                thickness=5,
                lineType=cv2.LINE_8,
            )
        else:
            rec_st = 1
            cv2.putText(
                image,
                text="recording...",
                org=(int(width // 2) - 70, int(height // 2)),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1.0,
                color=(0, 0, 255),
                thickness=5,
                lineType=cv2.LINE_8,
            )
            cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 10)

        cv2.imshow("display", image)

        cv2.waitKey(100)
        preset_count -= 1
        if preset_count == -fps * 3:
            break

    # 処理中の画面
    processing = np.zeros((int(height), int(width), 3), np.uint8)
    processing[:, :, 2] = 255
    cv2.putText(
        processing,
        text="processing...",
        org=(int(width // 2) - 70, int(height // 2)),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=3,
        lineType=cv2.LINE_8,
    )
    while 1:
        cv2.imshow("display", processing)
        cv2.waitKey(10)
        if proc_end:
            print("proc_end")
            break

    # 認識したルート音を表示
    root_disp = np.zeros((int(height), int(width), 3), np.uint8)
    root_disp[:, :, 2] = 255
    cv2.putText(
        root_disp,
        text="root is " + tone[root_idx],
        org=(int(width // 2) - 50, int(height // 2)),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=3,
        lineType=cv2.LINE_8,
    )

    cv2.imshow("display", root_disp)
    cv2.waitKey(500)

    ##############################

    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # model = cv2.bgsegm.createBackGroundSubtractorGMG()

    # model = cv2.createBackgroundSubtractorMOG2()

    # bg_img = cv2.imread("bg.jpg")
    # bg_w = bg_img.shape[1]
    # bg_h = bg_img.shape[0]

    prev_index_tip_r = [0, 0]  # 人差し指の先の座標(右)
    index_tip_r = [0, 0]  # [横, 縦]
    prev_index_tip_l = [0, 0]  # 人差し指の先の座標(左)
    index_tip_l = [0, 0]

    # 変位を記録するバッファ
    d_l = deque()
    d_r = deque()

    buff_size = int(fps / 2)
    # 1/2秒分の差分をとるバッファ
    for i in range(buff_size):
        d_l.append(0)
        d_r.append(0)

    # 左右の手が検知されているか [左, 右]
    detected = [0, 0]

    # 表示用
    up_st = [-1, -1, -1]
    down_st = [-1, -1, -1]

    org_lcr = [
        (int(width / 6), int(height / 2)),
        (int(width * 3 / 6), int(height / 2)),
        (int(width * 5 / 6), int(height / 2)),
    ]

    # for demo
    # fmt = cv2.VideoWriter_fourcc('m','p', '4','v')
    # writer = cv2.VideoWriter('./output/demo.mp4',fmt, fps, (int(width),int(height)))

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.flip(image, 1)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 背景をライブ会場に

            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # mask = model.apply(image)
            # dst = np.copy(bg_img)
            # image[mask == 0] = 0
            # dst = np.where(image==0, dst[int(bg_h-height):int(bg_h),int((bg_w-width)//2):int((bg_w-width)//2+width)], image)
            # image = dst

            # 指先の座標の更新
            if results.multi_hand_landmarks:  # 手が検知されているとき
                if len(results.multi_hand_landmarks) == 1:  # 手が1個のとき
                    if (
                        results.multi_handedness[0].classification[0].label == "Left"
                    ):  # 左手のみ
                        detected = [1, 0]
                        prev_index_tip_l = index_tip_l
                        index_tip_l = [
                            results.multi_hand_landmarks[0].landmark[8].x,
                            results.multi_hand_landmarks[0].landmark[8].y,
                        ]
                        if check_is_gu(results.multi_hand_landmarks[0].landmark):
                            is_gu_l += 1
                        else:
                            is_gu_l = 0

                    elif (
                        results.multi_handedness[0].classification[0].label == "Right"
                    ):  # 右手のみ
                        detected = [0, 1]
                        prev_index_tip_r = index_tip_r
                        index_tip_r = [
                            results.multi_hand_landmarks[0].landmark[8].x,
                            results.multi_hand_landmarks[0].landmark[8].y,
                        ]
                        if check_is_gu(results.multi_hand_landmarks[0].landmark):
                            is_gu_r += 1
                        else:
                            is_gu_r = 0

                if len(results.multi_hand_landmarks) == 2:
                    if (
                        results.multi_handedness[0].classification[0].label
                        == results.multi_handedness[1].classification[0].label
                    ):
                        if (
                            results.multi_handedness[0].classification[0].label
                            == "Left"
                        ):  # 左手のみ
                            detected = [1, 0]
                            prev_index_tip_l = index_tip_l
                            index_tip_l = [
                                results.multi_hand_landmarks[0].landmark[8].x,
                                results.multi_hand_landmarks[0].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[0].landmark):
                                is_gu_l += 1
                            else:
                                is_gu_l = 0

                        elif (
                            results.multi_handedness[0].classification[0].label
                            == "Right"
                        ):  # 右手のみ
                            detected = [0, 1]
                            prev_index_tip_r = index_tip_r
                            index_tip_r = [
                                results.multi_hand_landmarks[0].landmark[8].x,
                                results.multi_hand_landmarks[0].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[0].landmark):
                                is_gu_r += 1
                            else:
                                is_gu_r = 0

                    else:
                        if (
                            results.multi_handedness[0].classification[0].label
                            == "Left"
                        ):  # 左手→右手
                            prev_index_tip_l = index_tip_l
                            index_tip_l = [
                                results.multi_hand_landmarks[0].landmark[8].x,
                                results.multi_hand_landmarks[0].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[0].landmark):
                                is_gu_l += 1
                            else:
                                is_gu_l = 0

                            prev_index_tip_r = index_tip_r
                            index_tip_r = [
                                results.multi_hand_landmarks[1].landmark[8].x,
                                results.multi_hand_landmarks[1].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[1].landmark):
                                is_gu_r += 1
                            else:
                                is_gu_r = 0
                        elif (
                            results.multi_handedness[1].classification[0].label
                            == "Left"
                        ):  # 右手→左手
                            prev_index_tip_l = index_tip_l
                            index_tip_l = [
                                results.multi_hand_landmarks[1].landmark[8].x,
                                results.multi_hand_landmarks[1].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[1].landmark):
                                is_gu_l += 1
                            else:
                                is_gu_l = 0

                            prev_index_tip_r = index_tip_r
                            index_tip_r = [
                                results.multi_hand_landmarks[0].landmark[8].x,
                                results.multi_hand_landmarks[0].landmark[8].y,
                            ]
                            if check_is_gu(results.multi_hand_landmarks[0].landmark):
                                is_gu_r += 1
                            else:
                                is_gu_r = 0
                        detected = [1, 1]

                # 左手が検出されたとき
                if detected[0]:
                    check_l_c_r_l(index_tip_l)
                    draw_rec_l(image)
                    # バッファの更新
                    d_l.popleft()
                    if -0.05 < index_tip_l[1] - prev_index_tip_l[1] < 0.05:
                        d_l.append(0)
                    elif index_tip_l[1] - prev_index_tip_l[1] > 0:
                        d_l.append(-1)
                    else:
                        d_l.append(1)

                    cv2.circle(
                        image,
                        (int(index_tip_l[0] * width), int(index_tip_l[1] * height)),
                        30,
                        (255, 0, 0),
                        -1,
                    )

                # 右手が検出されたとき
                if detected[1]:
                    check_l_c_r_r(index_tip_r)
                    draw_rec_r(image)
                    # バッファの更新
                    d_r.popleft()
                    if -0.05 < index_tip_r[1] - prev_index_tip_r[1] < 0.05:
                        d_r.append(0)
                    elif index_tip_r[1] - prev_index_tip_r[1] > 0:
                        d_r.append(-1)
                    else:
                        d_r.append(1)

                    cv2.circle(
                        image,
                        (int(index_tip_r[0] * width), int(index_tip_r[1] * height)),
                        30,
                        (0, 255, 0),
                        -1,
                    )

            # 上下の動きを検知して、変化があればバッファをリセットする
            for i in range(3):
                if detected[0] and l_c_r_l == lcr[i]:
                    updown[i] = check_buff(d_l)
                    if updown[i] != 0:
                        reset_buff(d_l, buff_size)
                    if is_gu_l >= 5:
                        is_gu[i] = 1
                    else:
                        is_gu[i] = 0
                    # is_gu_l = 0
                if detected[1] and l_c_r_r == lcr[i]:
                    updown[i] = check_buff(d_r)
                    if updown[i] != 0:
                        reset_buff(d_r, buff_size)

                    if is_gu_r >= 5:
                        is_gu[i] = 1
                    else:
                        is_gu[i] = 0
                    # is_gu_r = 0

            # up / down を画面に表示
            for i in range(3):
                if updown[i] == 1:
                    up_st[i] = 0
                    print("up")
                if up_st[i] != -1:
                    cv2.putText(
                        image,
                        text="up",
                        org=org_lcr[i],
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(0, 0, 255),
                        thickness=5,
                        lineType=cv2.LINE_8,
                    )
                    up_st[i] += 1
                    if up_st[i] == fps:
                        up_st[i] = -1
                if updown[i] == -1:
                    down_st[i] = 0
                    print("down")
                if down_st[i] != -1:
                    cv2.putText(
                        image,
                        text="down",
                        org=org_lcr[i],
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(0, 0, 255),
                        thickness=5,
                        lineType=cv2.LINE_8,
                    )
                    down_st[i] += 1
                    if down_st[i] == fps:
                        down_st[i] = -1
                dia_idx[i] += updown[i]
                pitch_idx[i] = (
                    root_idx + int(dia_idx[i] // 7) * 12 + dia_to_chroma[dia_idx[i] % 7]
                )
                updown[i] = 0

            for i in range(3):
                if is_gu[i]:
                    cv2.putText(
                        image,
                        text="rest",
                        org=(org_lcr[i][0], int(org_lcr[i][1] / 2)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(15, 192, 252),
                        thickness=5,
                        lineType=cv2.LINE_8,
                    )
                else:
                    cv2.putText(
                        image,
                        text=tone[pitch_idx[i] % 12],
                        org=(org_lcr[i][0], int(org_lcr[i][1] / 2)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(15, 192, 252),
                        thickness=5,
                        lineType=cv2.LINE_8,
                    )

            # writer.write(image)
            cv2.imshow("display", image)
            if cv2.waitKey(33) & 0xFF == 27:
                cap_end = 1
                break
    cap.release()
    # writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # t2 = threading.Thread(target=showVideo)
    t1 = threading.Thread(target=man_audio)
    t2 = threading.Thread(target=showVideo)

    t1.start()
    t2.run()

    t1.join()
    t2.join()
