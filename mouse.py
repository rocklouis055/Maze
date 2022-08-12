# Taken from husano896's PR thread (slightly modified)
import pygame
from pygame.locals import *
import time
pygame.init()
screen = pygame.display.set_mode((1366, 600))
clock = pygame.time.Clock()

def a():
   c=0
   while True:
      for event in pygame.event.get():
            if event.type == QUIT:
               pygame.quit()
               return
      #print(pygame.mouse.get_pressed())
      print(pygame.mouse.get_pos())
      #print(pygame.mouse.get_rel())
      #time.sleep(1)
      #pygame.mouse.set_pos([500,500])
      c=1-c
      #pygame.mouse.set_visible(c)
      #pygame.mouse.get_visible()
      #print(pygame.mouse.get_focused())
      print(pygame.mouse.get_cursor())
# Execute game:
a()