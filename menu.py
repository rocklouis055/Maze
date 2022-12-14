from turtle import title
import pygame
import pygame_menu
import pygame_gui
import sys
from tkinter import *
from tkinter import messagebox
import cv2
import pygame
import numpy as np
from time import time
from PIL import Image
import time as tm
import matplotlib.pyplot as plt
def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float64)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result
def is_over(rect, pos):
    return True if rect.collidepoint(pos[0], pos[1]) else False
#INSIDE OF THE GAME LOOP
def is_over_circle(centre,rad,pos):
    dist=((centre[0]-pos[0])**2+(centre[1]-pos[1])**2)**0.5
    return True if dist<=rad else False

def chcolor(surface, color):
    """Fill all pixels of the surface with color, preserve transparency."""
    w, h = surface.get_size()
    r, g, b= color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))

pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")

def image(capimg,width,height,size,brect,bicon,brectc,biconc,bg):
    surface = pygame.display.set_mode([width,height])
    running =1
    f=1
    c=0
    pygame.init()
    font = pygame.font.Font('freesansbold.ttf', size+2)
    conretcolor=(255,255,255)
    conretclick=(0,0,0)
    msgcon = font.render("Continue", 1, conretcolor)
    conrect = msgcon.get_rect()
    conrect.center = (width//2,9*height//10)
    msgconclk = font.render("Continue", 1, conretclick)
    conrectclk = msgconclk.get_rect()
    conrectclk.center = conrect.center
    butboxcol=(0,0,0)
    butboxclk=(255,255,255)
    conbox=pygame.Rect(0,0,size*6,size*2)
    conbox.center=conrect.center
    while running:
        if f==1:
            pygame.init()
            font = pygame.font.Font('freesansbold.ttf', size+2)
            startloc=None
            endloc=None
            msgstart = font.render("Start :", 1, (255,255,255))
            msgend = font.render('End :',True, (255,255,255))
            startrect = msgstart.get_rect()
            endrect = msgend.get_rect()
            startrect.center = (1*width//6,9.5*height//10)
            endrect.center = (4.2*width//6,9.5*height//10)
            startbox=pygame.Rect(0,0,size*4.5,size*1.5)
            endbox=pygame.Rect(0,0,size*4.5,size*1.5)
            startbox.center=(startrect.center[0])+4.5*size,(startrect.center[1])
            endbox.center=endrect.center[0]+4.5*size,(endrect.center[1])
            msgsel = font.render("Select the Start and the End of the Maze :", 1, (0,0,0))
            selrect = msgsel.get_rect()
            selrect.center = (width//2,8.7*height//10)
            f+=1
        if f==2:
            surface.blit(bg,(0,0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and startloc is None and is_over(surf.get_rect(),pygame.mouse.get_pos()):
                    startloc=pygame.mouse.get_pos()
                    msgstartloc = font.render(",".join([str(i) for i in startloc]), 1, (0,0,0))
                    startlocrect = msgstartloc.get_rect()
                    startlocrect.center = startbox.center
                elif event.type == pygame.MOUSEBUTTONDOWN and endloc is None and is_over(surf.get_rect(),pygame.mouse.get_pos()):
                    endloc=pygame.mouse.get_pos()
                    msgendloc = font.render(",".join([str(i) for i in endloc]), 1, (0,0,0))
                    endlocrect = msgendloc.get_rect()
                    endlocrect.center = endbox.center
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(brect,pygame.mouse.get_pos()):
                    print("returning")
                    return (None,-1)
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(conbox,pygame.mouse.get_pos()):
                    f+=1
                    c=1
                    break
            if c:
                c=0
                continue
            capimg=cv2.resize(capimg,dsize=(5*height//6,5*width//6))
            surf = pygame.surfarray.make_surface(capimg)
            surface.blit(surf, (size*3,0))
            surface.blit(msgstart,startrect)
            surface.blit(msgend,endrect)
            pygame.draw.rect(surface,(255,255,255),startbox,0,5)
            pygame.draw.rect(surface,(255,255,255),endbox,0,5)
            pygame.draw.rect(surface,(255,255,255),((size*3,0),(5*width//6,5*height//6)),3,10)
            if not startloc is None:
                surface.blit(msgstartloc,startlocrect)
            if  not endloc is None:
                surface.blit(msgendloc,endlocrect)
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(bicon,brect)
            else:
                surface.blit(biconc,brectc)
            if not startloc is None and not endloc is None:
                if is_over(conbox,pygame.mouse.get_pos()):
                    surface.blit(msgconclk,conrectclk)
                    pygame.draw.rect(surface,butboxclk,conbox,3,7)
                else:
                    surface.blit(msgcon,conrect)
                    pygame.draw.rect(surface,(0,0,0),conbox,3,7)
            else:
                surface.blit(msgsel,selrect)
        if f==3:
            pass
        pygame.display.flip()
    return(None,+1)
def cam(width,height,size,link=0,flip=0,rot=0):
    surface = pygame.display.set_mode([width,height])
    #0 Is the built in camera
    e=(255,216,251)
    s=(79,40,74)
    array = get_gradient_3d(width, height, s,e, (1,1,1))
    img=Image.fromarray(np.uint8(array)).convert("RGBA")
    raw=img.tobytes("raw",'RGBA')
    bg = pygame.image.fromstring(raw, img.size, "RGBA")
    red=(255,0,0)
    running =1
    bgcolor=(79,45,74)
    surface.fill(bgcolor)
    f=1
    conretcolor=(0,0,0)
    conretclick=(255,255,255)
    font = pygame.font.Font('freesansbold.ttf', size*2)
    msgcon = font.render("Continue", 1, conretcolor)
    msgret = font.render('Retake', True, conretcolor)
    conrect = msgcon.get_rect()
    retrect = msgret.get_rect()
    conrect.center = (1*1280//6,5*720//6)
    retrect.center = (5*1280//6,5*720//6)
    msgconclk = font.render("Continue", 1, conretclick)
    msgretclk = font.render('Retake',True, conretclick)
    conrectclk = msgconclk.get_rect()
    retrectclk = msgretclk.get_rect()
    conrectclk.center = (1*1280//6,5*720//6)
    retrectclk.center = (5*1280//6,5*720//6)
    butboxcol=(0,0,0)
    butboxclk=(255,255,255)
    conbox=pygame.Rect(0,0,size*10,size*3)
    retbox=pygame.Rect(0,0,size*8,size*3)
    conbox.center=conrect.center
    retbox.center=retrect.center
    msg1string='On the next screen, take'
    msg2string='the picture of the maze.'
    e,s=0,0
    startloc=pygame.mouse.get_pos()
    msgstartloc = font.render("       ", 1, (0,0,0))
    startlocrect = msgstartloc.get_rect()
    startlocrect.center =(0,0)
    endloc=pygame.mouse.get_pos()
    msgendloc = font.render("       ", 1, (0,0,0))
    endlocrect = msgendloc.get_rect()
    endlocrect.center =(0,0)
    c=0
    while running:
        s=time()
        print(1/(s-e))
        e=time()
        if f==1:
            pygame.init()
            font = pygame.font.Font('freesansbold.ttf', size+2)
            mcolor=(0,0,0)
            msg1 = font.render(msg1string, 1, mcolor)
            msg2 = font.render(msg2string, True, mcolor)
            oktext=font.render('OK!', 0, (0,255,0))
            okbox=pygame.Rect(width//2,height//2,size*3,size*1.5)
            biconr=pygame.image.load('./back.png')
            bicon=pygame.transform.scale(biconr,(3*size,3*size))
            biconc=bicon.copy()
            chcolor(biconc,(0,0,0))
            brect=bicon.get_rect()
            brectc=biconc.get_rect()
            brect.center=(width//8,height//8)
            brectc.center=(width//8,height//8)
            textRect1 = msg1.get_rect()
            textRect2 = msg2.get_rect()
            okRect = oktext.get_rect()
            textRect1.center = (width//2,height//2-size)
            textRect2.center = (width//2,height//2+size)
            okRect.center = (width//2,height//2+3*size)
            okbox.center=okRect.center
            f+=1
            continue
        if f==2:
            surface.blit(bg, (0, 0))
            if is_over(okbox,pygame.mouse.get_pos()):
                pygame.draw.rect(surface,(80,80,80),okbox,20)
            else:
                pygame.draw.rect(surface,(127,127,127),okbox,20)
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(bicon,brect)
            else:
                surface.blit(biconc,brectc)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type==pygame.MOUSEBUTTONDOWN and is_over(okbox,pygame.mouse.get_pos()):
                    f+=1
                    c=1
                    break
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(brect,pygame.mouse.get_pos()):
                    f=1
                    msg1string="Image not taken, take"
                    msg2string="again on the next screen"
                    pygame.quit()
                    surface = pygame.display.set_mode([width,height])
                    cv2.destroyAllWindows()
                    return
            if c:
                c=0
                continue
            surface.blit(msg1, textRect1)
            surface.blit(msg2, textRect2)
            surface.blit(oktext, okRect)
        
            
        if f==3:
            
            cap = cv2.VideoCapture(link) 
            #cap = cv2.VideoCapture(2) #iriun cam
            #cap = cv2.VideoCapture("http://192.168.43.1:8080/video") #ip webcam
            #cap = cv2.VideoCapture("https://192.168.43.1:4343/video?1280x720") #droid cam

            #Gets fps of your camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!
            #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #print(width,height)

            fps = cap.get(cv2.CAP_PROP_FPS)
            print("fps:", fps)
            #If your camera can achieve 60 fps
            #Else just have this be 1-30 fps
            cap.set(cv2.CAP_PROP_FPS, 30)
            f+=1
            pygame.quit()
        if f==4:
            surface = pygame.display.set_mode([1280,720])
            
            success, frame = cap.read()
            if flip:
                frame=np.flip(frame,1)
            if rot:
                frame = np.rot90(frame,rot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            surf = pygame.surfarray.make_surface(frame)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    running = False
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over_circle((1280/2,9*720/10),30,pygame.mouse.get_pos()):
                    pygame.draw.circle(surface,(0,200,0),(1280/2,9*720/10),30,0)
                    pygame.display.flip()
                    c=1
                    break
                elif event.type==pygame.MOUSEBUTTONUP and is_over_circle((1280/2,9*720/10),30,pygame.mouse.get_pos()):
                    f+=1
                    capimg=frame
                    cap.release()
                    break
                    c=1
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(brect,pygame.mouse.get_pos()):
                    f=1
                    msg1string="Image not taken, take"
                    msg2string="again on the next screen"
                    pygame.quit()
                    surface = pygame.display.set_mode([width,height])
                    cap.release()
                    break
                    c=1
            if c:
                c=0
                continue
            surface.blit(surf, (0,0))
            if is_over_circle((1280/2,9*720/10),30,pygame.mouse.get_pos()):
                pygame.draw.circle(surface,(0,0,200),(1280/2,9*720/10),30,0)
            else:
                pygame.draw.circle(surface,(200,0,0),(1280/2,9*720/10),30,0)
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(bicon,brect)
            else:
                surface.blit(biconc,brectc)
            pygame.draw.circle(surface,(200,0,0),(1280/2,9*720/10),40,3)
        if f==5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type==pygame.MOUSEBUTTONDOWN and (is_over(brect,pygame.mouse.get_pos()) or is_over(retbox,pygame.mouse.get_pos())):
                    f=3
                    startloc,endloc=None,None
                    c=1
                    break
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(conbox,pygame.mouse.get_pos()):
                    f+=1
                    pygame.quit()
                    surface = pygame.display.set_mode([width,height])
                    c=1
                    break
            if c:
                c=0
                continue
            surf = pygame.surfarray.make_surface(capimg)
            surface.blit(surf, (0,0))
            if is_over(conbox,pygame.mouse.get_pos()):
                surface.blit(msgconclk,conrectclk)
                pygame.draw.rect(surface,butboxclk,conbox,3,17)
            else:
                surface.blit(msgcon,conrect)
                pygame.draw.rect(surface,butboxcol,conbox,3,17)
            if is_over(retbox,pygame.mouse.get_pos()):
                surface.blit(msgretclk,retrectclk)
                pygame.draw.rect(surface,butboxclk,retbox,3,17)
            else:
                surface.blit(msgret,retrect)
                pygame.draw.rect(surface,butboxcol,retbox,3,17)
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(bicon,brect)
            else:
                surface.blit(biconc,brectc)
        if f==6:
            a,b=image(capimg,width,height,size,brect,bicon,brectc,biconc,bg)
            if b==-1:
                f=3
            else:
                break
            continue
        pygame.display.flip()
    pygame.quit()


pygame.init()
surface = pygame.display.set_mode((1000, 600))

# def set_mode(value, difficulty):
#     # Do the job here !
#     pass

"""######### AUTO GAME PLAY START HERE #########"""
def genmaze_solve():
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AUTO MAZEGEN | SOLVER")

    manager = pygame_gui.UIManager((1000, 600))
    clock = pygame.time.Clock()
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((WIDTH/2, HEIGHT/2), (100, 50)), manager=manager,                                           object_id='#hight_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((WIDTH/2, (HEIGHT+(HEIGHT/4.5))/2), (100, 50)), manager=manager,                      object_id='#width_maze')
    

    def show_user_input(hight_maze,width_maze):
        print("in")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.fill("black")

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            loading_text = pygame.font.SysFont("HELVETICA", 40).render("Loading...", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            loading_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/4))/2, (HEIGHT+(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(loading_text, loading_text_rect)

            clock.tick(60)

            pygame.display.update()

    def get_user_input():
        l=[]
        while True:
            UI_REFRESH_RATE = clock.tick(60)/1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#hight_maze'):
                    l.append(event.text)
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#width_maze'):
                    l.append(event.text)
                    show_user_input(int(l[0]),int(l[1]))
            
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            SCREEN.fill("black")

            manager.draw_ui(SCREEN)
            new_text = pygame.font.SysFont("HELVETICA", 40).render("Enter the Size of the MAZE:", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, (HEIGHT-(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)

            height_text = pygame.font.SysFont("HELVETICA", 35).render("Height :", True, "white")
            height_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/8))/2, (HEIGHT+(HEIGHT/12))/2))
            SCREEN.blit(height_text, height_text_rect)
            
            width_text = pygame.font.SysFont("HELVETICA", 35).render("Width :", True, "white")
            width_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/8))/2, (HEIGHT+(HEIGHT/3))/2))
            SCREEN.blit(width_text, width_text_rect)
            clock.tick(60)
            pygame.display.update()
    get_user_input()

"""######### MAIN GAME PLAY START HERE #########"""
def start_the_game():
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AUTO MAZEGEN | SOLVER")

    manager = pygame_gui.UIManager((1000, 600))
    clock = pygame.time.Clock()
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((WIDTH/2, HEIGHT/2), (100, 50)), manager=manager,                                           object_id='#hight_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((WIDTH/2, (HEIGHT+(HEIGHT/4.5))/2), (100, 50)), manager=manager,                      object_id='#width_maze')
    

    def show_user_input(hight_maze,width_maze):
        print("in")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.fill("black")

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            loading_text = pygame.font.SysFont("HELVETICA", 40).render("Loading...", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            loading_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/4))/2, (HEIGHT+(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(loading_text, loading_text_rect)

            clock.tick(60)

            pygame.display.update()

    def get_user_input():
        l=[]
        while True:
            UI_REFRESH_RATE = clock.tick(60)/1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#hight_maze'):
                    l.append(event.text)
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#width_maze'):
                    l.append(event.text)
                    show_user_input(int(l[0]),int(l[1]))
            
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            SCREEN.fill("black")

            manager.draw_ui(SCREEN)
            new_text = pygame.font.SysFont("HELVETICA", 40).render("Enter the Size of the MAZE:", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, (HEIGHT-(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)

            height_text = pygame.font.SysFont("HELVETICA", 35).render("Height :", True, "white")
            height_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/8))/2, (HEIGHT+(HEIGHT/12))/2))
            SCREEN.blit(height_text, height_text_rect)
            
            width_text = pygame.font.SysFont("HELVETICA", 35).render("Width :", True, "white")
            width_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/8))/2, (HEIGHT+(HEIGHT/3))/2))
            SCREEN.blit(width_text, width_text_rect)
            clock.tick(60)
            pygame.display.update()
    get_user_input()

def web_input():
    pass

def file_input():
    pass


"""############ TEXT WORK DONE HERE  ################"""
def text_input():
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TEXT_INPUT MAZE | SOLVER")

    manager = pygame_gui.UIManager((1000, 600))
    
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH+100)/4, (HEIGHT+(HEIGHT/2))/8), (100, 50)), manager=manager,object_id='#hight_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH+200)/2, (HEIGHT+(HEIGHT/2))/8), (100, 50)), manager=manager,object_id='#width_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH-200)/2, ((HEIGHT-50)/2)), (550, 300)), manager=manager,object_id='#matrix_maze')

    clock = pygame.time.Clock()

    def show_user_input(hight_maze,width_maze,matrix_maze):
        print("in")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.fill("black")

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            matrix_text = pygame.font.SysFont("HELVETICA", 40).render(f"MAZE : {matrix_maze}", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            matrix_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/4))/2, (HEIGHT+(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(matrix_text, matrix_text_rect)

            clock.tick(60)

            pygame.display.update()

    def get_user_input():
        l=[]
        while True:
            UI_REFRESH_RATE = clock.tick(60)/1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#hight_maze'):
                    l.append(int(event.text))
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#width_maze'):
                    l.append(int(event.text))
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#matrix_maze'):
                    l.append(event.text.split(' '))
                    print(l)
                    print(len(l[2]))
                    print(
)
                    if(l[0]*l[1]==len(l[2])):
                        show_user_input(int(l[0]),int(l[1]),l[2])
                    else:
                        # text_box = pygame_gui.TextBox(html_text="<effect id=test>ERROR: Enter the parameters Again!!!</effect>",relative_rect=pygame.Rect(100, 100, 200, 50),manager=manager)
                        # text_box.set_active_effect(pygame_gui.TEXT_EFFECT_FADE_OUT, effect_tag='test')
                        # matrix_text = pygame.font.SysFont("HELVETICA", 35).render("ERROR: Enter the parameters Again!!!", True, "white")
                        # matrix_text_rect = new_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, HEIGHT/2))
                        # SCREEN.blit(matrix_text, matrix_text_rect)
                        Tk().wm_withdraw() #to hide the main window
                        messagebox.showinfo('MessagePop','ERROR: Enter the parameters Again!!!')
                        get_user_input()
            
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            SCREEN.fill("black")

            manager.draw_ui(SCREEN)
            new_text = pygame.font.SysFont("HELVETICA", 40).render("Enter the Size of the MAZE:", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, (HEIGHT-(HEIGHT/4))/16))
            SCREEN.blit(new_text, new_text_rect)

            height_text = pygame.font.SysFont("HELVETICA", 35).render("Height :", True, "white")
            height_text_rect = new_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, (HEIGHT+(HEIGHT/2))/6.5))
            SCREEN.blit(height_text, height_text_rect)
            
            width_text = pygame.font.SysFont("HELVETICA", 35).render("Width :", True, "white")
            width_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/2.7))/2, (HEIGHT+(HEIGHT/2))/6.5))
            SCREEN.blit(width_text, width_text_rect)

            matrix_text = pygame.font.SysFont("HELVETICA", 35).render("Enter The Matrix :", True, "white")
            matrix_text_rect = new_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, HEIGHT/2))
            SCREEN.blit(matrix_text, matrix_text_rect)

            clock.tick(60)
            pygame.display.update()
    get_user_input()

def camera_input():
    cam(700,500,20,1,0,1)


"""#########   _MAIN_ PROGRAM STARTS HERE    #########"""
default='Louise :) '
# menu = pygame_menu.Menu(f'Welcome {default}', 1300, 700,
#                        theme=pygame_menu.themes.THEME_BLUE)
# font=pygame_menu.font.FONT_HELVETICA

mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0),widget_font=('HELVETICA'), # transparent background
                widget_font_color=(0, 0, 0)
            )
mytheme.title_background_color=(191, 0, 255)
mytheme.title_font_color=(255,255,255)



myimage = pygame_menu.baseimage.BaseImage(
    './img9.jpg',drawing_mode=pygame_menu.baseimage.IMAGE_MODE_FILL,
    drawing_offset=(0,0)
)
mytheme.background_color = myimage
menu = pygame_menu.Menu(f'Welcome {default}', 1000, 600,
                       theme=mytheme )

# menu.add.text_input('Enter Your Name :', default)
# menu.add.selector('Mode :', [('Computer', 1), ('User', 2)], onchange=set_mode)
menu.add.button('> Auto Generator and Solver',genmaze_solve)
menu.add.button('> Play the Maze Game', start_the_game)
menu.add.button('> Image from WEB',web_input)
menu.add.button('> File input',file_input)
menu.add.button('> Text input',text_input)
menu.add.button('> Camera input',camera_input)
menu.add.button('> Quit', pygame_menu.events.EXIT)

menu.mainloop(surface)