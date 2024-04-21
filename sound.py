import time
import pygame.mixer

pygame.mixer.init(frequency=44100)
pygame.mixer.music.load("abunaiyokiwotsukete_01.wav")

pygame.mixer.music.play(1)

time.sleep(3)

pygame.mixer.music.stop()

