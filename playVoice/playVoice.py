import time
import pygame.mixer

# 開始処理
pygame.mixer.init(frequency = 44100)

# 再生対象のファイル指定
pygame.mixer.music.load("abunaiyokiwotsukete_01.wav")

# 何回再生を繰り返すか(-1 = 無限)
pygame.mixer.music.play(1)

# 再生時間(sec)
time.sleep(3)

# 終了処理
pygame.mixer.music.stop()
