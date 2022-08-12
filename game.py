from PIL import Image
import numpy as np
import math
import re
from time import time
from random import choice
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



n,m=map(int,input("Enter the dimention M*N :").split())

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
k,path,t=genMaze(m=m,n=n)
maze1=Maze(k)
if(maze1.check()):
  temp=maze1.path()

a,b=genimage(maze=k,solved=temp,minpixel=5,offset=1,solvedpath=[255,255,255],wall=[47, 37, 74],path=[255,255,255],end=[200,10,10],save=1)

a,b=genimage(name="final",maze=k,solved=temp,minpixel=5,offset=1,solvedpath=[230, 21, 153],wall=[47, 37, 74],path=[255,255,255],end=[200,10,10],save=1)

size=m*2-1+4
#print(k)
# Import and initialize the pygame library
import pygame
from pygame.locals import *
pygame.init()
clock = pygame.time.Clock()
# Set up the drawing window
screen = pygame.display.set_mode([1366,728])
i=0
width = 720
height = 720
size=width/(m-1+4)
# Run until the user asks to quit
running = True
j=1
del t2
t2=[]
for i in range(1,len(t)):
  if abs(2*t[i-1][0]-2*t[i][0])>1:
    for x in range(min(2*t[i-1][0],2*t[i][0]),max(2*t[i-1][0],2*t[i][0])+1):
      t2.append([x,2*t[i][1]])
  if abs(2*t[i-1][1]-2*t[i][1])>1:
    for y in range(min(2*t[i-1][1],2*t[i][1]),max(2*t[i-1][1],2*t[i][1])+1):
      t2.append([2*t[i][0],y])
t2=[]
for i in range(1,len(t)):
  if abs(2*t[i-1][0]-2*t[i][0])>1:
    for x in range(min(2*t[i-1][0],100000000000),max(-1,2*t[i][0])+1):
      t2.append([x,2*t[i][1]])
  if abs(2*t[i-1][1]-2*t[i][1])>1:
    for y in range(min(2*t[i-1][1],100000000000),max(-1,2*t[i][1])+1):
      t2.append([2*t[i][0],y])
import time
font = pygame.font.SysFont(None, 55)
pygame.mixer.init()
pygame.mixer.music.load('music.mp3')
def text_screen(text, color, x, y):
    screen_text = font.render(text, True, color)
    screen.blit(screen_text, [x, y]) 
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
time.sleep(3)
while running:
    # Did the user click the window close button?
#     k,i=t[j]
    for event in pygame.event.get():
        
        if event.type == pygame.KEYDOWN:
          print(event.key)
          if event.key == pygame.K_w:
              print("Move the character forwards")
          elif event.key == pygame.K_s:
              print("Move the character backwards")
          elif event.key == pygame.K_a:
              print("Move the character left")
          elif event.key == pygame.K_d:
              print("Move the character right")
        if event.type == pygame.QUIT:
          running = False
    pygame.display.flip()
    pygame.display.update()
running=0
while running:
    # Did the user click the window close button?
#     k,i=t[j]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            running = False
    # Fill the background with white
    # for m in range(1,len(t)):
#     for x in range(2*t[j-1][0],2*t[j][0]+1):
#       pygame.draw.rect(screen, (255,255,255), [1+size*(x+2), 1+size*(k), size+2, size+2] )
#     for y in range(2*t[j-1][1],2*t[j][1]+1):
#       pygame.draw.rect(screen, (255,255,255), [1+size*(k), 1+size*(y+2), size+2, size+2] )
    #~~~~~~~
    # for x in range(2*t[j-1][0],2*t[j][0]+1):
    #   pygame.draw.rect(screen, (255,255,255), [(1+size*(x+2)), (1+size*(t[j][1]+2)), size+2, size+2] )
    # for y in range(2*t[j-1][1],2*t[j][1]+1):
    #   pygame.draw.rect(screen, (255,255,255), [(1+size*(t[j][0]+2)), (1+size*(y+2)), size+2, size+2] )
    #~~~~~~~
    y,x=t2[j]
    #if(t2[j-1][0]-y==1):
     #   pygame.draw.rect(screen, (255,255,255), [y*size+2*size,x*size+2*size,size,size]) 
#    if(t2[j-1][0]-y==-1):
 #       pygame.draw.rect(screen, (255,255,255), [t2[j][1]*size+2*size,t2[j][0]*size+2*size,size,size])
  #  if(t2[j-1][1]-x==1):
   #     pygame.draw.rect(screen, (255,255,255), [y*size+2*size,x*size+size,size,size])
    #if(t2[j-1][1]-x==-1):
     #   pygame.draw.rect(screen, (255,255,255), [t2[j][1]*size+2*size,t2[j-1][0]*size+2*size,size,size])
    # Draw a solid blue circle in the center
    # Flip the display
    pygame.draw.rect(screen, (255,255,255), [x*size+2*size,y*size+2*size,size+1,size+1])
    j+=1
    # if j==len(maze1.nstack):
    if j==len(t2):
        #screen.fill((0,0,0))
        break
    pygame.display.flip()
    pygame.display.update()
    clock.tick(2000)
# Done! Time to quit.
#bg_img = pygame.image.load('maze.png')
#bg_img = pygame.transform.scale(bg_img,(width,height))
#window.blit(bg_img,(0,0))
running = 0
j=1
while running:
    # Did the user click the window close button?
    k,i=maze1.nstack[j]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            running = False
    
    # Fill the background with white
    pygame.draw.rect(screen, (255,0,255), [size*(i+2), size*(k+2), size+1, size+1] )
    # Draw a solid blue circle in the center
    # Flip the display
    j+=1
    if j==len(maze1.nstack):
        
        screen.fill((0,0,0))
        time.sleep(3)
        text_screen("You Won ", (255,0,0), 100, 350)

        break
        
    pygame.display.flip()
    pygame.display.update()
    clock.tick(500)
# Done! Time to quit.
pygame.quit()