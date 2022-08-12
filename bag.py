import pygame
print(pygame.__version__)
pygame.init()
screen=pygame.display.set_mode([pygame.display.Info().current_w,pygame.display.Info().current_h])
multi=pygame.display.Info().current_w/1000
print(multi)
pygame.draw.rect(screen, (255,0,255), [0,0,160*4*multi,90*4*multi] )
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.update()
