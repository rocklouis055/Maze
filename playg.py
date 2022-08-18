from tracemalloc import start
from PIL import Image
import numpy as np
import math
import re
from time import time,sleep

from random import choice
import pygame
from pygame.locals import *
def chcolor(surface, color):
    """Fill all pixels of the surface with color, preserve transparency."""
    w, h = surface.get_size()
    r, g, b= color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))
class Maze:
  def __init__(self,matrix):
    self.matrix=matrix
    self.row=len(matrix)
    self.col=len(matrix[0])
    self.visited=np.zeros((self.row, self.col))
    self.count2=0
    self.count3=0
    self.valid=0
    self.loc=[0,0,0,0]
    self.jstack=[]
    self.nstack=[]
    self.dist=0
  def check(self):
    for i in range(self.row):
      for j in range(self.col):
        if (self.matrix[i][j]==2):
          self.count2+=1
          self.loc[0],self.loc[1]=i,j
        if(self.matrix[i][j]==3): 
          self.count3+=1
          self.loc[2],self.loc[3]=i,j
    if(self.count2==1 and self.count3==1):
      self.valid=1
      print("Valid Maze")
    return(self.valid)
      
  def path(self):
    i=self.loc[0]
    j=self.loc[1]
    self.len=1
    self.count=0
    self.up=0
    self.down=0
    self.left=0
    self.right=0
    while(1):
      self.up=0
      self.down=0
      self.left=0
      self.right=0
      
      if(i==self.loc[2] and j==self.loc[3]):
        print("Found path!")
        break
      elif(i<self.row-1 and self.matrix[i+1][j] and self.visited[i+1][j]!=1):
        self.down+=1
        #i+=1 
      elif(i>0 and self.matrix[i-1][j] and self.visited[i-1][j]!=1):
        self.up+=1
        #i-=1
      elif(j<self.col-1 and self.matrix[i][j+1] and self.visited[i][j+1]!=1):
        self.right+=1
        #j+=1
      elif(j>0 and self.matrix[i][j-1] and self.visited[i][j-1]!=1):
        self.left+=1
        #j-=1
      self.count=self.up+self.down+self.right+self.left
      self.nstack.append([i,j])
      if(self.count>0):
        self.jstack.append([i,j,self.count,len(self.nstack)])
      if(self.right==1):
        j+=1
        self.visited[i][j]=1
      elif(self.down==1):
        i+=1
        self.visited[i][j]=1
      elif(self.up==1):
        i-=1
        self.visited[i][j]=1
      elif(self.left==1):
        j-=1
        self.visited[i][j]=1
      else:
        i,j,self.count,t=self.jstack.pop()
        del self.nstack[t-1:]
    self.temp=[]
    for i in range(self.row):
      self.temp.append([0]*self.col)
    for i,j in self.nstack:
      self.temp[i][j]=1
    return(self.temp)
  def block(self,i,j,count):
    if(i>0 and self.visited[i-1][j]==count):
      self.visited[i-1][j]=0
    elif(i<self.col-1 and self.visited[i+1][j]==count):
      self.visited[i+1][j]=0
    elif(j>0 and self.visited[i][j-1]==count):
      self.visited[i][j-1]=0
    elif(j<self.row-1 and self.visited[i][j+1]==count):
      self.visited[i][j+1]=0
def genimage(maze,solved=None,mul=None,name='maze',minpixel=5,resolution=None,row=None,col=None,solvedpath=[0,255,0],path=[255,255,255],wall=[0,0,0],start=[0,255,0],end=[0,0,255],offset=1,save=0):
  if row==None:
    row=len(maze)
  if col==None:
    col=len(maze[0])
  if mul is None:
    if resolution is None:
      resolution=minpixel*col
      mul=minpixel
    elif resolution<col*minpixel:
      print("Cant generate image!,Change parameters.")
      return(None,None)
    else:
      mul=math.ceil(resolution/col)
  if offset==1:
    row+=4
    col+=4
    a=np.zeros([row*mul,col*mul,3],dtype='uint8')
    for i in range(mul):
      for j in range(row*mul):
        a[j][i]=a[j][(col-1)*mul+i]=wall
      for j in range(col*mul):
        a[i][j]=a[(row-1)*mul+i][j]=wall
  else:
    a=np.zeros([row*mul,col*mul,3],dtype='uint8')
  
  for i in range(2*mul*offset,(row-2*offset)*mul):
    for j in range(2*mul*offset,(col-2*offset)*mul):
      k=maze[(i-2*mul*offset)//mul][(j-2*mul*offset)//mul]
      l=solved[(i-2*mul*offset)//mul][(j-2*mul*offset)//mul]
      if l==1 and k<2:
        a[i][j]=solvedpath
      elif k==1:
        a[i][j]=path
      elif k==2:
        a[i][j]=start
      elif k==3:
        a[i][j]=end
      else:
        a[i][j]=wall
  img = Image.fromarray(a)
  if(save):
    img.save(name+'.png')
  return(img,a)



initial=[0,0]
t2=[]
def genMaze(n=4,m=4,initial=[0,0],final=None,infivalue=[2,3],save=0,name="UnsolvedMaze"):
  if n*m//4<2:
    print("Cant generate image! Change Parameters.")
    return
  n,m=n//2,m//2
  l=np.ones([m,n],dtype="int")
  f=np.zeros([m*2-1,n*2-1],dtype="int")
  s=[]
  i=initial[0]
  j=initial[1]
  k=[]
  t=[]
  while(1):
    k=[]
    t.append([i,j])
    if(i-1>=0 and l[i-1][j]==1):k.append([i-1,j])
    if(j-1>=0 and l[i][j-1]==1):k.append([i,j-1])
    if(i+1<m and l[i+1][j]==1):k.append([i+1,j])
    if(j+1<n and l[i][j+1]==1):k.append([i,j+1])
    l[i][j]=0
    if len(k)>0:
      s.append([[i,j],k])
      i,j=choice(k)
      t2.append([i,j])
    if len(k)==0:
      a=s.pop()
      if(a[0]==initial):
        break
      else:
        i,j=a[0]
  pathMaze=[]
  for i in range(1,len(t)):
    for x in range(2*t[i-1][0],2*t[i][0]+1):
      f[x][2*t[i][1]]=1
      pathMaze.append([x,2*t[i][1]])
    for y in range(2*t[i-1][1],2*t[i][1]+1):
      f[2*t[i][0]][y]=1
      pathMaze.append([2*t[i][0],y])
  f[0][0],f[-1][-1]=infivalue
  return(f,pathMaze,t)
def is_over(rect, pos):
    return True if rect.collidepoint(pos[0], pos[1]) else False
#INSIDE OF THE GAME LOOP
def is_over_circle(centre,rad,pos):
    dist=((centre[0]-pos[0])**2+(centre[1]-pos[1])**2)**0.5
    return True if dist<=rad else False
def text_screen(screen,font,text, color, x, y):
        screen_text = font.render(text, True, color)
        screen.blit(screen_text, [x, y]) 
def game(m,n,height=720,width=720,speed=None,bg=None):
    k,path,t=genMaze(m=m,n=n)
    maze1=Maze(k)
    if(maze1.check()):
        temp=maze1.path()
    size=m*n/2
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([width,height])
    i=0
    size=width/(m-1+4)
    # Run until the user asks to quit
    running = True
    j=1
    t2=[]
    for i in range(1,len(t)):
        if abs(2*t[i-1][0]-2*t[i][0])>1:
            for x in range(min(2*t[i-1][0],2*t[i][0]),max(2*t[i-1][0],2*t[i][0])+1):
                t2.append([x,2*t[i][1]])
        if abs(2*t[i-1][1]-2*t[i][1])>1:
            for y in range(min(2*t[i-1][1],2*t[i][1]),max(2*t[i-1][1],2*t[i][1])+1):
                t2.append([2*t[i][0],y])
    # t2=[]
    # for i in range(1,len(t)):
    #     if abs(2*t[i-1][0]-2*t[i][0])>1:
    #         for x in range(min(2*t[i-1][0],100000000000),max(-1,2*t[i][0])+1):
    #             t2.append([x,2*t[i][1]])
    #     if abs(2*t[i-1][1]-2*t[i][1])>1:
    #         for y in range(min(2*t[i-1][1],100000000000),max(-1,2*t[i][1])+1):
    #             t2.append([2*t[i][0],y])
    
    font = pygame.font.SysFont(None, int(size*2))
    pygame.mixer.init()
    #pygame.mixer.music.load('music.mp3')
    
    pygame.draw.rect(screen, (47,37,74), [0,0,700,700] )
    # print(maze1.nstack)
    j=0
    print(pygame.display.Info())
    border=(47,37,74)
    window = pygame.display.set_mode((width,height))
    pygame.draw.rect(screen, border, [0,0,size,height])
    pygame.draw.rect(screen, border, [0,0,width,size])
    pygame.draw.rect(screen, border, [0,height-size,width,size+4])
    pygame.draw.rect(screen, border, [width-size,0,size+4,height])
    pygame.draw.rect(screen, border, [size*2,size*2,width-size*4,height-size*4])
    msgcol=(255,0,0)
    if bg is None:
        bg=pygame.image.load("./color_gradient.jpg")
    ct=time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        y,x=t2[j]
        pygame.draw.rect(screen, (255,255,255), [x*size+2*size,y*size+2*size,size+1,size+1])
        j+=1
        # if j==len(maze1.nstack):
        if j==len(t2):
            #screen.fill((0,0,0))
            break
        pygame.display.flip()
        pygame.display.update()
        clock.tick(speed)
    running = 1
    j=1
    x,y=0,0
    running = True
    print(len(k),len(k[0]))
    covered=(200,200,255)
    current=(50,255,50)
    timer=(255,255,255)
    fonttime = pygame.font.SysFont(None, int(size*.2))
    fontmsg = pygame.font.SysFont(None, int(size*2))
    # screen_text = font.render(text, True, color)
    #     screen.blit(screen_text, [x, y]) 
    it=time()
    f=1
    flag=1
    endcol=(0,0,255)
    startcol=(255,0,0)
    msgcol=(200,0,0)
    conretcolor=(0,0,0)
    conretclick=(255,255,255)
    msgcon = font.render("Continue to the Menu", 1, conretcolor)
    msgret = font.render('Back to the Gane', True, conretcolor)
    conrect = msgcon.get_rect()
    retrect = msgret.get_rect()
    conrect.center = (1.5*width//6,5*height/6)
    retrect.center = (4.5*width//6,5*height/6)
    msgconclk = font.render("Continue to the Menu", 1, conretclick)
    msgretclk = font.render('Back to the Gane',True, conretclick)
    conrectclk = msgconclk.get_rect()
    retrectclk = msgretclk.get_rect()
    conrectclk.center = (1.5*width//6,5*height/6)
    retrectclk.center = (4.5*width//6,5*height/6)
    butboxcol=(0,0,0)
    butboxclk=(255,255,255)
    conbox=pygame.Rect(0,0,(height/20)*8,(height/20)*2)
    retbox=pygame.Rect(0,0,(height/20)*8,(height/20)*2)
    conbox.center=conrect.center
    retbox.center=retrect.center

    biconr=pygame.image.load('./back.png')
    bicon=pygame.transform.scale(biconr,(3*size,3*size))
    biconc=bicon.copy()
    chcolor(biconc,(0,0,0))
    brect=bicon.get_rect()
    brectc=biconc.get_rect()
    brect.center=(width//8,height//8)
    brectc.center=(width//8,height//8)
    c=0
    while running:
        if flag==1:
            ct=time()-it
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                flag+=1
                pygame.image.save(screen,"./.cache/temp1.png")
                continue
            if y+1<len(k[0]) and (keys[pygame.K_d] or keys[pygame.K_RIGHT]) and (k[x][y+1]!=0):
                pygame.draw.rect(screen, covered, (size*(y+2),size*(x+2), size+1, size+1))
                y+=1
            
            if y-1>=0 and (keys[pygame.K_a] or keys[pygame.K_LEFT ]) and (k[x][y-1]!=0):
                pygame.draw.rect(screen, covered, (size*(y+2),size*(x+2), size+1, size+1))
                y-=1
            if x+1<len(k) and (keys[pygame.K_s] or keys[pygame.K_DOWN ]) and (k[x+1][y]!=0):
                pygame.draw.rect(screen, covered, (size*(y+2),size*(x+2), size+1, size+1))
                x+=1
            if x-1>=0 and (keys[pygame.K_w] or keys[pygame.K_UP   ]) and (k[x-1][y]!=0):
                pygame.draw.rect(screen, covered, (size*(y+2),size*(x+2), size+1, size+1))
                x-=1
            if x==len(k)-1 and y==len(k[0])-1:
                if f:
                    wt=ct
                    f=0
                screen.fill((0,0,0))
                textmsg=fontmsg.render("You Won in %02d:%02d:%02d"%((wt//(60**2))%60,(wt//60)%60,wt%60),1,msgcol)
                screen.blit(textmsg,(0,0))
                break
                f+=1
                
            pygame.draw.rect(screen, current, (size*(y+2),size*(x+2), size+1, size+1))
            pygame.draw.rect(screen, startcol, (size*2,size*2, size+1, size+1))
            pygame.draw.rect(screen, endcol, (size*n,size*m, size+1, size+1))
                #screen.blit(bg,(0,0))
        if flag==2:
            screen.blit(bg,(0,0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return
                elif event.type==pygame.MOUSEBUTTONDOWN and (is_over(brect,pygame.mouse.get_pos()) or is_over(retbox,pygame.mouse.get_pos()) or event.type==pygame.K_ESCAPE):
                    flag-=1
                    screen.blit(pygame.image.load("./.cache/temp1.png"),(0,0))
                    c=1
                    break
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(conbox,pygame.mouse.get_pos()):
                    pygame.quit()
                    
                    screen = pygame.display.set_mode([width,height])
                    c=1
                    return
            if c:
                c=0
                continue
            if is_over(conbox,pygame.mouse.get_pos()):
                screen.blit(msgconclk,conrectclk)
                pygame.draw.rect(screen,butboxclk,conbox,3,17)
            else:
                screen.blit(msgcon,conrect)
                pygame.draw.rect(screen,butboxcol,conbox,3,17)
            if is_over(retbox,pygame.mouse.get_pos()):
                screen.blit(msgretclk,retrectclk)
                pygame.draw.rect(screen,butboxclk,retbox,3,17)
            else:
                screen.blit(msgret,retrect)
                pygame.draw.rect(screen,butboxcol,retbox,3,17)
            if is_over(brect,pygame.mouse.get_pos()):
                screen.blit(bicon,brect)
            else:
                screen.blit(biconc,brectc)
            
        pygame.display.update()
        pygame.draw.rect(screen,border,(width/2-size*0.5,0,size*5,size))
        clock.tick(20)
    # Done! Time to quit.
    pygame.quit()
#game(20,20,400,400)
def auto(m,n,height,width,speed,bg=None):
    k,path,t=genMaze(m=m,n=n)
    maze1=Maze(k)
    if(maze1.check()):
      temp=maze1.path()
    size=m*2-1+4
    biconr=pygame.image.load('./back.png')
    bicon=pygame.transform.scale(biconr,(int(size*0.8),int(size*0.8)))
    biconc=bicon.copy()
    chcolor(biconc,(0,0,0))
    brect=bicon.get_rect()
    brectc=biconc.get_rect()
    brect.center=(width//8,height//8)
    brectc.center=(width//8,height//8)
    if bg is None:
      bg=pygame.image.load("./color_gradient.jpg")
    
    

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([width,height])
    i=0
    size=width/(m-1+4)
    # Run until the user asks to quit
    running = True
    j=1
    t2=[]
    for i in range(1,len(t)):
        if abs(2*t[i-1][0]-2*t[i][0])>1:
            for x in range(min(2*t[i-1][0],2*t[i][0]),max(2*t[i-1][0],2*t[i][0])+1):
                t2.append([x,2*t[i][1]])
        if abs(2*t[i-1][1]-2*t[i][1])>1:
            for y in range(min(2*t[i-1][1],2*t[i][1]),max(2*t[i-1][1],2*t[i][1])+1):
                t2.append([2*t[i][0],y])
    # t2=[]
    # for i in range(1,len(t)):
    #     if abs(2*t[i-1][0]-2*t[i][0])>1:
    #         for x in range(min(2*t[i-1][0],100000000000),max(-1,2*t[i][0])+1):
    #             t2.append([x,2*t[i][1]])
    #     if abs(2*t[i-1][1]-2*t[i][1])>1:
    #         for y in range(min(2*t[i-1][1],100000000000),max(-1,2*t[i][1])+1):
    #             t2.append([2*t[i][0],y])
    
    font = pygame.font.SysFont(None, int(size*3))
    pygame.mixer.init()
    #pygame.mixer.music.load('music.mp3')
    pygame.draw.rect(screen, (47,37,74), [0,0,700,700] )
    # print(maze1.nstack)
    j=0
    print(pygame.display.Info())
    border=(47,37,74)
    window = pygame.display.set_mode((width,height))
    pygame.draw.rect(screen, border, [0,0,size,height])
    pygame.draw.rect(screen, border, [0,0,width,size])
    pygame.draw.rect(screen, border, [0,height-size,width,size+4])
    pygame.draw.rect(screen, border, [width-size,0,size+4,height])
    pygame.draw.rect(screen, border, [size*2,size*2,width-size*4,height-size*4])
    msgcol=(255,0,0)
    
    ct=time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
        y,x=t2[j]
        pygame.draw.rect(screen, (255,255,255), [x*size+2*size,y*size+2*size,size+1,size+1])
        j+=1
        # if j==len(maze1.nstack):
        if j==len(t2):
            #screen.fill((0,0,0))
            break
        pygame.display.flip()
        pygame.display.update()
        clock.tick(speed)
    running = 1
    j=1
    f=1
    while running:
        if f==1:
            temp,i=maze1.nstack[j]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    running = False
                    return
            pygame.draw.rect(screen, (255,0,255), [size*(i+2), size*(temp+2), size+1, size+1] )
            j+=1
            if j==len(maze1.nstack):
                f+=1
                ct=time()-ct
        if f==2:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    running = False
                    pygame.quit()

                    return
                elif event.type==pygame.MOUSEBUTTONDOWN and (is_over(brect,pygame.mouse.get_pos()) or is_over(retbox,pygame.mouse.get_pos()) or event.type==pygame.K_ESCAPE):
                    return
            
            screen.fill((0,0,0))
            text=font.render(("Animation finished in : %02d:%02d:%02d"%((ct//(60**2))%60,(ct//60)%60,ct%60)),1,msgcol)
            text_rect = text.get_rect(center=(width/2, height/2))
            screen.blit(bg,(0,0))
            screen.blit(text, text_rect)
            if is_over(brect,pygame.mouse.get_pos()):
                screen.blit(bicon,brect)
            else:
                screen.blit(biconc,brectc)
            
        pygame.display.flip()
        clock.tick(speed/4)
def mazeanim(m,n,mat,height,width,size,speed,bg=None):
    tmpwidth=int((width/height)*(size*n)+1)
    tmpheight=int((height/width)*(size*m)+1)
    
    if tmpheight>(height):
        height=int((height))
        width=tmpwidth
    elif tmpwidth>(width):
        width=int((width))
        height=tmpheight
    maze1=Maze(mat)
    if(maze1.check()):
      path=maze1.path()
    
    size=m*2-1+4
    biconr=pygame.image.load('./back.png')
    bicon=pygame.transform.scale(biconr,(int(size*1.5),int(size*1.5)))
    biconc=bicon.copy()
    chcolor(biconc,(0,0,0))
    brect=bicon.get_rect()
    brectc=biconc.get_rect()
    brect.center=(width//8,height//8)
    brectc.center=(width//8,height//8)
    if bg is None:
      bg=pygame.image.load("./color_gradient.jpg")
    
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([width,height])
    i=0
    size=width/(m-1+4)
    # Run until the user asks to quit
    running = True
    j=1
    t2=[]
    
    font = pygame.font.SysFont(None, int(size*1.5))
    pygame.mixer.init()
    #pygame.mixer.music.load('music.mp3')
    # pygame.draw.rect(screen, (47,37,74), [0,0,700,700] )
    # print(maze1.nstack)
    j=0
    print(pygame.display.Info())
    border=(47,37,74)
    window = pygame.display.set_mode((width,height))
    pygame.draw.rect(screen,border,pygame.Rect(0,0,height,width),int(size+1))
    # pygame.draw.rect(screen, border, [0,0,size,height])
    # pygame.draw.rect(screen, border, [0,0,width,size])
    # pygame.draw.rect(screen, border, [0,height-size,width,size+4])
    # pygame.draw.rect(screen, border, [width-size,0,size+4,height])
    # pygame.draw.rect(screen, border, [size*2,size*2,width-size*4,height-size*4])
    msgcol=(255,0,0)
    
    ct=time()
    colblock=(50,0,200)
    colpath=(255,255,255)
    colstart=(255,0,0)
    colend=(0,255,0)
    for i in range(len(mat)):
      for j in range(len(mat[0])):
        if mat[i][j]==2:
          pygame.draw.rect(screen, colstart, [i*size+2*size,j*size+2*size,size+1,size+1])
        elif mat[i][j]==3:
          pygame.draw.rect(screen, colend, [i*size+2*size,j*size+2*size,size+1,size+1])
        elif mat[i][j]==1:
          pygame.draw.rect(screen, colpath, [i*size+2*size,j*size+2*size,size+1,size+1])
        else:
          pygame.draw.rect(screen, colblock, [i*size+2*size,j*size+2*size,size+1,size+1])
    pygame.display.flip()     
    running = 1
    j=1
    f=1
    while running:
        if f==1:
            temp,i=maze1.nstack[j]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    running = False
                    return
            pygame.draw.rect(screen, (255,0,255), [size*(temp+2), size*(i+2), size+1, size+1] )
            j+=1
            if j==len(maze1.nstack):
                f+=1
                ct=time()-ct
        if f==2:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    running = False
                    pygame.quit()

                    return
                elif event.type==pygame.MOUSEBUTTONDOWN and (is_over(brect,pygame.mouse.get_pos())):
                    return
            
            screen.fill((0,0,0))
            text=font.render(("Animation finished in : %02d:%02d:%02d"%((ct//(60**2))%60,(ct//60)%60,ct%60)),1,msgcol)
            text_rect = text.get_rect(center=(width/2, height/2))
            screen.blit(bg,(0,0))
            screen.blit(text, text_rect)
            if is_over(brect,pygame.mouse.get_pos()):
                screen.blit(bicon,brect)
            else:
                screen.blit(biconc,brectc)
            
        pygame.display.flip()
        clock.tick(speed/4)

#game(50,50,720,720,50*50/2)
#auto(50,50,720,720,50*50/2)
m,n=20,10
k,path,t=genMaze(m=m,n=n)
mazeanim(m,n,k,720,720,n/25,m*n/2)
