import pygame
import os
pygame.init()
size=20
screen = pygame.display.set_mode((700,500))
# create the display surface object  
# of specific dimension..e(500, 500).  
def text_screen(text, color, x, y):
    screen_text = pygame.font.Font("Auckland Hills.ttf",50).render(text, True, color)
    screen.blit(screen_text, [x, y]) 
def welcome(height,width,size):
    welc=(255,255,255)
    
    if(not os.path.exists("userinfo.txt")):
        with open("userinfo.txt", "w") as f:
            f.write("User")
    with open("userinfo.txt", "r") as f:
        user =f.read()
    bg_img = pygame.image.load('754632.webp')
    bg_img = pygame.transform.scale(bg_img,(width,height))
    running=1
    screen.blit(bg_img,(0,0))
    user_text = ''

    # create rectangle
    input_rect = pygame.Rect(200, 200, 140, 32)

    # color_active stores color(lightskyblue3) which
    # gets active when input box is clicked by user
    color_active = pygame.Color('lightskyblue3')

    # color_passive store color(chartreuse4) which is
    # color of input box.
    color_passive = pygame.Color('chartreuse4')
    color = color_passive

    active = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if input_rect.collidepoint(event.pos):
                active = True
            else:
                active = False

        if event.type == pygame.KEYDOWN:

			# Check for backspace
            if event.key == pygame.K_BACKSPACE:

				# get text input from 0 to -1 i.e. end.
                user_text = user_text[:-1]

			# Unicode standard is used for string
			# formation
            else:
                user_text += event.unicode
        pygame.draw.rect(screen, color, input_rect)
        screen.fill((255, 255, 255))

        if active:
            color = color_active
        else:
            color = color_passive

        pygame.draw.rect(screen, color, input_rect)

        text_surface = pygame.font.Font.render(user_text, True, (255, 255, 255))
	
	# render at position stated in arguments
        screen.blit(text_surface, (input_rect.x+5, input_rect.y+5))
	
	# set width of textfield so that text cannot get
	# outside of user's text input
        input_rect.w = max(100, text_surface.get_width()+10)
	
	# display.flip() will update only a portion of the
	# screen to updated, not full area
        pygame.display.flip()
        text_screen(f"Welcome" ,welc,width/4,height/7)
        text_screen(f"{user}" ,welc,2*width/4,2*height/7)
        pygame.display.flip()
welcome(500,700,20)