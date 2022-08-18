from tkinter import CENTER
import cv2
import pygame
import numpy as np
from time import time
from PIL import Image,ImageOps
import time as tm
import math
from random import choice
from warnings import warn
import heapq
from distutils import text_file
from turtle import title
import os
import platform
import mimetypes
import webbrowser
import shutil
#import wget
import requests
import pyperclip
import pygame
import pygame_menu
import pygame_gui
import sys
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
pygame.init()
username=""
WIDTH, HEIGHT = 1000, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.display.set_mode((1000, 600))
class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement = False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          # return return_path(current_node)       
        
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None
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
  # def path(self):
  #   i=self.loc[0]
  #   j=self.loc[1]
  #   self.len=1
  #   self.count=0
  #   self.up=0
  #   self.down=0
  #   self.left=0
  #   self.right=0
  #   while(1):
  #     self.up=0
  #     self.down=0
  #     self.left=0
  #     self.right=0
  #     #print(self.jstack)
  #     if(i==self.loc[2] and j==self.loc[3]):
  #       print("Found path!")
  #       break
  #     elif(i<self.row-1 and self.matrix[i+1][j] and self.visited[i+1][j]!=1):
  #       self.down+=1
  #       #i+=1 
  #     elif(i>0 and self.matrix[i-1][j] and self.visited[i-1][j]!=1):
  #       self.up+=1
  #       #i-=1
  #     elif(j<self.col-1 and self.matrix[i][j+1] and self.visited[i][j+1]!=1):
  #       self.right+=1
  #       #j+=1
  #     elif(j>0 and self.matrix[i][j-1] and self.visited[i][j-1]!=1):
  #       self.left+=1
  #       #j-=1
  #     self.count=self.up+self.down+self.right+self.left
  #     self.nstack.append([i,j])
  #     if(self.count>0):
  #       self.jstack.append([i,j,self.count,len(self.nstack)])
  #     if(self.right==1):
  #       j+=1
  #       self.visited[i][j]=1
  #     elif(self.down==1):
  #       i+=1
  #       self.visited[i][j]=1
  #     elif(self.up==1):
  #       i-=1
  #       self.visited[i][j]=1
  #     elif(self.left==1):
  #       j-=1
  #       self.visited[i][j]=1
  #     else:
  #       i,j,self.count,t=self.jstack.pop()
  #       del self.nstack[t-1:]
  #   self.temp=[]
  #   for i in range(self.row):
  #     self.temp.append([0]*self.col)
  #   for i,j in self.nstack:
  #     self.temp[i][j]=1
  #   return(self.temp)
  # def block(self,i,j,count):
  #   if(i>0 and self.visited[i-1][j]==count):
  #     self.visited[i-1][j]=0
  #   elif(i<self.col-1 and self.visited[i+1][j]==count):
  #     self.visited[i+1][j]=0
  #   elif(j>0 and self.visited[i][j-1]==count):
  #     self.visited[i][j-1]=0
  #   elif(j<self.row-1 and self.visited[i][j+1]==count):
  #     self.visited[i][j+1]=0      
  def path(self):
    i=self.loc[0]
    j=self.loc[1]
    self.len=1
    self.count=0
    self.up=0
    self.down=0
    self.left=0
    self.right=0
    d=[]
    b=[]
    while(1):
      self.up=0
      self.down=0
      self.left=0
      self.right=0
      d={}
      # (li-i)**2+(lj-j)**2
      #print(i<self.row-1 and self.matrix[i+1][j] and self.visited[i+1][j]!=1,i>0 and self.matrix[i-1][j] and self.visited[i-1][j]!=1,j<self.col-1 and self.matrix[i][j+1] and self.visited[i][j+1]!=1,j>0 and self.matrix[i][j-1] and self.visited[i][j-1]!=1)
      if(i==self.loc[2] and j==self.loc[3]):
        print("Found path!")
        break
      if(i<self.row-1 and self.matrix[i+1][j] and self.visited[i+1][j]!=1):
        self.down+=1
        #i+=1 
        te=(math.fabs(self.loc[0]-(i))+math.fabs(self.loc[1]-j))+math.fabs(self.loc[2]-(i+1))+math.fabs(self.loc[3]-j)
        if not te in d:
          d[te]=[i+1,j]
        else:
          d[te+1]=[i+1,j]

      elif(j<self.col-1 and self.matrix[i][j+1] and self.visited[i][j+1]!=1):
        self.right+=1
        #j+=1
        te=math.fabs(self.loc[0]-(i))+math.fabs(self.loc[1]-(j))+math.fabs(self.loc[2]-(i))+math.fabs(self.loc[3]-(j+1))
        if not te in d:
          d[te]=[i,j+1]
        else:
          d[te+1]=[i,j+1]

      elif(i>0 and self.matrix[i-1][j] and self.visited[i-1][j]!=1):
        self.up+=1
        #i-=1
        te=math.fabs(self.loc[0]-(i))+math.fabs(self.loc[1]-j)+math.fabs(self.loc[2]-(i-1))+math.fabs(self.loc[3]-j)
        if not te in d:
          d[te]=[i-1,j]
        else:
          d[te+1]=[i-1,j]

      elif(j>0 and self.matrix[i][j-1] and self.visited[i][j-1]!=1):
        self.left+=1
        #j-=1
        te=math.fabs(self.loc[0]-(i))+math.fabs(self.loc[1]-(j))+math.fabs(self.loc[2]-(i))+math.fabs(self.loc[3]-(j-1))
        if not te in d:
          d[te]=[i,j-1]
        else:
          d[te+1]=[i,j-1]
      self.count=self.up+self.down+self.right+self.left
      self.nstack.append([i,j])
      b={}
      for i1 in sorted(d.keys(),reverse=1):
        b[i1]=d[i1]
      if len(b.keys())>1:
        print("  ",b)
      if(self.count>0):
        self.jstack.append([i,j,self.count,len(self.nstack)])
      # #print(b,d)
      # if(self.right==1 and (b[list(b.keys())[-1]])==(i,j+1)):
      #   j+=1
      #   self.visited[i][j]=1
      # elif(self.down==1 and (b[list(b.keys())[-1]])==(i+1,j)):
      #   i+=1
      #   self.visited[i][j]=1
      # elif(self.up==1 and (b[list(b.keys())[-1]])==(i-1,j)):
      #   i-=1
      #   self.visited[i][j]=1
      # elif(self.left==1 and (b[list(b.keys())[-1]])==(i,j-1)):
      #   j-=1
      #   self.visited[i][j]=1
      if self.count>0:
        i,j=b[list(b.keys())[-1]]
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



# n,m=map(int,input("Enter the dimention M*N :").split())

# initial=[0,0]
# t2=[]
# def genMaze(n=4,m=4,initial=[0,0],final=None,infivalue=[2,3],save=0,name="UnsolvedMaze"):
#   if n*m//4<2:
#     print("Cant generate image! Change Parameters.")
#     return
#   n,m=n//2,m//2
#   l=np.ones([m,n],dtype="int")
#   f=np.zeros([m*2-1,n*2-1],dtype="int")
#   s=[]
#   i=initial[0]
#   j=initial[1]
#   k=[]
#   t=[]
#   while(1):
#     k=[]
#     t.append([i,j])
#     if(i-1>=0 and l[i-1][j]==1):k.append([i-1,j])
#     if(j-1>=0 and l[i][j-1]==1):k.append([i,j-1])
#     if(i+1<m and l[i+1][j]==1):k.append([i+1,j])
#     if(j+1<n and l[i][j+1]==1):k.append([i,j+1])
#     l[i][j]=0
#     if len(k)>0:
#       s.append([[i,j],k])
#       i,j=choice(k)
#       t2.append([i,j])
#     if len(k)==0:
#       a=s.pop()
#       if(a[0]==initial):
#         break
#       else:
#         i,j=a[0]
#   pathMaze=[]
#   for i in range(1,len(t)):
#     for x in range(2*t[i-1][0],2*t[i][0]+1):
#       f[x][2*t[i][1]]=1
#       pathMaze.append([x,2*t[i][1]])
#     for y in range(2*t[i-1][1],2*t[i][1]+1):
#       f[2*t[i][0]][y]=1
#       pathMaze.append([2*t[i][0],y])
#   f[0][0],f[-1][-1]=infivalue
#   return(f,pathMaze,t)



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

def image(capimg,width,height,size,flag,bg=None,f2=0):
    startloc=endloc=None
    if bg is None:
        bg=pygame.image.load("./color_gradient.jpg")
    biconr=pygame.image.load('./back.png')
    bicon=pygame.transform.scale(biconr,(3*size,3*size))
    biconc=bicon.copy()
    chcolor(biconc,(0,0,0))
    brect=bicon.get_rect()
    brectc=biconc.get_rect()
    brect.center=(width//8,height//8)
    brectc.center=(width//8,height//8)
    if flag:
        c=np.flipud(capimg)
        c = np.rot90(c,3)
        c=cv2.flip(c,0)
        c=cv2.cvtColor(c,cv2.COLOR_BGR2RGB)
        cv2.imwrite("a1.png",c)
        cv2.imwrite("a1.jpg",c)
        cv2.imwrite("a1.jpeg",c)
        cv2.imwrite("a1.bmp",c)
        cv2.imwrite("a1.webp",c)
    #ar=np.zeros(capimg.shape)
    print(np.shape(capimg))
    #capimg=cv2.cvtColor(capimg,cv2.COLOR_BGR2HSV)
    # rw,rh=np.shape(capimg)[0],np.shape(capimg)[1]
    # a=np.zeros([rw,rh,3])
    # for i in range(rw):
    #     for j in range(rh):
    #         a[i][j]=capimg[i][j]
    # capimg=a
    # arr=np.zeros([rw,rh])
    # #print(arr)
    # for i in range(rw):
    #     for j in range(rh):
    #         s=(sum(capimg[i][j])/3)
    #         if s>200:
    #             v=1
    #         else:
    #             v=0
    #         arr[i][j]=v
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
    #capimg=np.rot90(capimg,2) #change
    print(capimg.size,capimg.shape)
    imgwid,imghei=capimg.shape[0],capimg.shape[1]
    print("correct ratio",imgwid/imghei,5*width/6,5*height/6)
    cv2.imwrite("a1.png",capimg)
    tmpimgwid=int((imgwid/imghei)*(5*height)/6+1)
    tmpimghei=int((imghei/imgwid)*(5*width)/6+1)
    print("temp",tmpimgwid,tmpimghei,tmpimgwid/tmpimghei)
    print("ck 1",tmpimgwid/(5*height/6))
    print("ck 2",(5*width/6)/tmpimghei)
    if tmpimghei>(5*height)/6:
        imghei=int((5*height)/6)
        imgwid=tmpimgwid
    elif tmpimgwid>(5*width)/6:
        imgwid=int((5*width)/6)
        imghei=tmpimghei
    print("mod",imgwid,imghei,imgwid/imghei)
    print("correct",capimg.shape[0]/capimg.shape[1],capimg.shape[1]/capimg.shape[0])
    capimg=cv2.resize(capimg,dsize=(imghei,imgwid))
    cv2.imwrite("a2.png",capimg)
    print(capimg.shape[0]/capimg.shape[1],capimg.shape[1]/capimg.shape[0])
    #capimg=cv2.resize(capimg,dsize=(5*height//6,5*width//6))
    #cv2.imwrite("a3.png",capimg)
    #print(capimg.shape[0]/capimg.shape[1],capimg.shape[1]/capimg.shape[0])
    #print(capimg)
    #capimg=cv2.flip(cv2.cvtColor(capimg,cv2.COLOR_BGR2RGB),1)
    print(capimg[0][0],capimg[0][-1],capimg[-1][-1],capimg[-1][0])
    a=np.zeros([imgwid,imghei,3])
    for i in range(imgwid):
        for j in range(imghei):
            a[i][j]=capimg[i][j]
    capimg=a.astype(np.uint8)
    arr=np.zeros([imgwid,imghei])
    arr2=np.zeros([imgwid,imghei])
    #print(arr)
    for i in range(imgwid):
        for j in range(imghei):
            s=(sum(capimg[i][j])/3)
            if s>200:
                v=1
            else:
                v=0
            arr[i][j]=v
            arr2[i][j]=1-v
    #print(arr)
   
    image2 = Image.fromarray(arr.astype(np.uint8))
    image2.save("a.png")
    print(capimg[0][0],capimg[0][-1],capimg[-1][-1],capimg[-1][0])
    
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
            surf = pygame.surfarray.make_surface(capimg)
            rect=surf.get_rect()
            rect.center=(width//2+1,5*height/12)
            surface.blit(bg,(0,0))
        if f==2:
            surface.blit(bg,(0,0))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and startloc is None and is_over(rect,pygame.mouse.get_pos()):
                    startloc=(pygame.mouse.get_pos()[0]-rect.x,pygame.mouse.get_pos()[1]-rect.y)
                    msgstartloc = font.render(",".join([str(i) for i in startloc]), 1, (0,0,0))
                    startlocrect = msgstartloc.get_rect()
                    startlocrect.center = startbox.center
                elif event.type == pygame.MOUSEBUTTONDOWN and endloc is None and is_over(rect,pygame.mouse.get_pos()):
                    endloc=(pygame.mouse.get_pos()[0]-rect.x,pygame.mouse.get_pos()[1]-rect.y)
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
            
            
            surface.blit(surf, (surf.get_rect(center=rect.center)))
            surface.blit(msgstart,startrect)
            surface.blit(msgend,endrect)
            pygame.draw.rect(surface,(255,255,255),startbox,0,5)
            pygame.draw.rect(surface,(255,255,255),endbox,0,5)
            pygame.draw.rect(surface,(255,255,255),rect,3,10)
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if f2:
                startloc=[imgwid-startloc[0],imghei-startloc[1]]
                endloc=[imgwid-endloc[0],imghei-endloc[1]]
            # arr[int(startloc[0]*rw/rh)][int(startloc[1]*rh/rw)]=2
            # arr[int(endloc[0]*rw/rh)][int(endloc[1]*rh/rw)]=3
            arr[int(startloc[0])][int(startloc[1])]=2
            arr[int(endloc[0])][int(endloc[1])]=3
            print(arr)
            print(capimg[0][0],capimg[0][-1],capimg[-1][-1],capimg[-1][0])

            maze1=Maze(arr)
            if(maze1.check()):
                temp=maze1.path()
            # temp=astar(arr2,startloc,endloc,1)
            # print("found path")
            a,b=genimage(maze=arr,solved=temp,minpixel=5,offset=1,solvedpath=[255,255,255],wall=[47, 37, 74],path=[255,255,255],end=[200,10,10],save=1)

            a,b=genimage(name="final",maze=arr,solved=temp,minpixel=5,offset=1,solvedpath=[230, 21, 153],wall=[47, 37, 74],path=[255,255,255],end=[200,10,10],save=1)

            return(0,0)
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
    font = pygame.font.Font('freesansbold.ttf', (720//20))
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
    conbox=pygame.Rect(0,0,(720//20)*7,(720//20)*2)
    retbox=pygame.Rect(0,0,(720//20)*5,(720//20)*2)
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
        #print(1/(s-e))
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
            #If your camera can achieve 6 fps
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
            pygame.quit()
            a,b=image(capimg,width,height,size,1,bg,f2=1)
            if b==-1:
                f=3
            else:
                break
            continue
        pygame.display.flip()
    pygame.quit()

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



def configure():
    file=open("configure.txt","r")
    temp=file.readline()
# def set_mode(value, difficulty):
#     # Do the job here !
#     pass
def is_over(rect, pos):
    return True if rect.collidepoint(pos[0], pos[1]) else False
def chcolor(surface, color):
    """Fill all pixels of the surface with color, preserve transparency."""
    w, h = surface.get_size()
    r, g, b= color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))


"""######### AUTO GAME PLAY START HERE #########"""
def genmaze_solve():
    pygame.init()
    WIDTH, HEIGHT = 700,500
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AUTO MAZEGEN | SOLVER")

    manager = pygame_gui.UIManager((700,500))
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
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        mainmenu(username)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc) # paint to screen
            # pygame.display.flip()

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            loading_text = pygame.font.SysFont("HELVETICA", 40).render("Loading...", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            loading_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/4))/2, (HEIGHT+(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(loading_text, loading_text_rect)
            pygame.display.update()
            tm.sleep(3)
            auto(hight_maze,width_maze,720,720,hight_maze*width_maze/2)
            break

            clock.tick(60)

           

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
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        mainmenu(username)
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

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
    pygame.display.set_caption("MAZEGAME | PLAYABLE")

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
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        get_user_input()

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))

            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            loading_text = pygame.font.SysFont("HELVETICA", 40).render("Loading...", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            loading_text_rect = new_text.get_rect(center=((WIDTH+(WIDTH/4))/2, (HEIGHT+(HEIGHT/4))/2))
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(loading_text, loading_text_rect)

            clock.tick(60)

            pygame.display.update()
            tm.sleep(3)
            game(hight_maze,width_maze,720,720,hight_maze*width_maze/2)
            break

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
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        mainmenu(username)
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

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


"""############### WEB WORK DONE HERE  #################"""
def web_input():
    def download(url: str,name, dest_folder: str):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)  # create folder if it does not exist
        response = requests.get(url)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        print(extension)

        filename = url.split('/')[-1].replace(" ", "_")
        if f"{extension}" not in filename:  # be careful with file names
            filename=name+f"{extension}"
        file_path = os.path.join(dest_folder, filename)

        r = requests.get(url, stream=True)
        if r.ok:
            print("saving to", os.path.abspath(file_path))
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        else:  # HTTP status code 4XX/5XX
            print("Download failed: status code {}\n{}".format(r.status_code, r.text))
    
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LINK MAZEGEN | SOLVER")

    manager = pygame_gui.UIManager((1000, 600))
    clock = pygame.time.Clock()
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH-200)/2, (HEIGHT+(HEIGHT/2))/8), (350, 50)), manager=manager,object_id='#web_maze_name')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH-200)/2, ((HEIGHT-50)/2)), (550, 50)), manager=manager,object_id='#web_maze')

    def show_user_input(link_maze):
        print("in")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        get_user_input()

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

            new_text = pygame.font.SysFont("HELVETICA", 40).render(f"Maze URL Given: {link_maze}", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            SCREEN.blit(new_text, new_text_rect)

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
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_object_id == '#web_maze_name'):
                    l.append(event.text)
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_object_id == '#web_maze'):
                    l.append(event.text)
                    download(f"{l[1]}",l[0],dest_folder="mini")
                    show_user_input(l[1])
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        mainmenu(username)

            
                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

            manager.draw_ui(SCREEN)

            name_text = pygame.font.SysFont("HELVETICA", 35).render("Name :", True, "white")
            name_text_rect = name_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, (HEIGHT+(HEIGHT/2))/6.5))
            SCREEN.blit(name_text, name_text_rect)

            web_text = pygame.font.SysFont("HELVETICA", 35).render("Enter The Link :", True, "white")
            web_text_rect = web_text.get_rect(center=(((WIDTH/2)+(WIDTH/14))/2, HEIGHT/2))
            SCREEN.blit(web_text, web_text_rect)
            clock.tick(60)
            pygame.display.update()
    get_user_input()


"""############ FILE WORK DONE HERE  ################"""
def file_input():
    Tk().wm_withdraw() #to hide the main window
    messagebox.showinfo('MessagePop','You have to select the file as an input here')
    
    def openFile():
        l=["jpg", "jpeg", "bmp", "webp", "png"]
        filepath = filedialog.askopenfilename(initialdir="C:\\Users\\",
                                          title="Select the file",
                                          filetypes= (("text files","*.txt"),
                                          ("image files","*.png .*jpg .*jpeg .*webp .*bmp")))    
                                        
        if ".txt" in filepath:
            file = open(filepath,'r')
            m,n=file.readline().split(" ")
            # tm.sleep(3)
            auto(m,n,720,720,m*n/2)
            
            file.close()
        else:
            print(filepath)

    button = Button(text="Open",command=openFile())
    button.pack()


"""############ TEXT WORK DONE HERE  ################"""
def text_input():
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TEXT_INPUT MAZE | SOLVER")

    manager = pygame_gui.UIManager((1000, 600))
    
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH+100)/4, (HEIGHT+(HEIGHT/2))/8), (100, 50)), manager=manager,object_id='#hight_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH+200)/2, (HEIGHT+(HEIGHT/2))/8), (100, 50)), manager=manager,object_id='#width_maze')
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH-200)/2, ((HEIGHT-50)/2)), (550, 50)), manager=manager,object_id='#matrix_maze')

    clock = pygame.time.Clock()

    def show_user_input(hight_maze,width_maze,matrix_maze):
        print("in")
        size=HEIGHT/25
        font = pygame.font.SysFont(None, int(size))
        msgcol=(200,0,0)
        conretcolor=(0,0,0)
        conretclick=(255,255,255)
        msgcon = font.render("Open Maze", 1, conretcolor)
        msgret = font.render('Reload', True, conretcolor)
        conrect = msgcon.get_rect()
        retrect = msgret.get_rect()
        conrect.center = (1.5*WIDTH//6,9*HEIGHT/10)
        retrect.center = (4.5*WIDTH//6,9*HEIGHT/10)
        msgconclk = font.render("Open Maze", 1, conretclick)
        msgretclk = font.render('Reload',True, conretclick)
        conrectclk = msgconclk.get_rect()
        retrectclk = msgretclk.get_rect()
        conrectclk.center = (1.5*WIDTH//6,9*HEIGHT/10)
        retrectclk.center = (4.5*WIDTH//6,9*HEIGHT/10)
        butboxcol=(0,0,0)
        butboxclk=(255,255,255)
        conbox=pygame.Rect(0,0,(HEIGHT/20)*8,(HEIGHT/20))
        retbox=pygame.Rect(0,0,(HEIGHT/20)*8,(HEIGHT/20))
        conbox.center=conrect.center
        retbox.center=retrect.center
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        return
                elif event.type==pygame.MOUSEBUTTONDOWN and is_over(conbox,pygame.mouse.get_pos()):
                    m=hight_maze
                    n=width_maze
                    mat=np.zeroes((m,n))
                    k=0
                    for i in range(hight_maze):
                        for j in range(width_maze):
                            mat[i][j]=(int(matrix_maze[k]))
                        k+=1
                    mazeanim(m,n,mat,720,720,n/25,m*n/2)
            
            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

            new_text = pygame.font.SysFont("HELVETICA", 35).render(f"Maze Size Given: {hight_maze}X{width_maze}", True, "white")
            matrix_maze_text = pygame.font.SysFont("HELVETICA", 35).render(f"MAZE :", True, "white")
            new_text_rect = new_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, (HEIGHT+(HEIGHT/2))/12))
            matrix_maze_text_rect = new_text.get_rect(center=(((WIDTH/2)+(WIDTH/5))/2, (HEIGHT+450+((HEIGHT)/2))/12))
            k=0
            nm=[]
            for i in range(hight_maze):
                a=[]
                for j in range(width_maze):
                    a.append(matrix_maze[k])
                    k+=1
                nm.append(a)
            
            w=((WIDTH/2)+(WIDTH/5)+250)/2
            h=(HEIGHT+600+((HEIGHT)/2))/12
            temp=w
            for i in range(hight_maze):
                for j in range(width_maze):
                    matrix_text = pygame.font.SysFont("HELVETICA", 25).render(f"{nm[i][j]}", True, "white")
                    matrix_text_rect = new_text.get_rect(center=(w,h))
                    w+=40
                    SCREEN.blit(matrix_text, matrix_text_rect)
                w=temp
                h+=35
            if is_over(conbox,pygame.mouse.get_pos()):
                SCREEN.blit(msgconclk,conrectclk)
                pygame.draw.rect(SCREEN,butboxclk,conbox,3,7)
            else:
                SCREEN.blit(msgcon,conrect)
                pygame.draw.rect(SCREEN,butboxcol,conbox,3,7)
            SCREEN.blit(new_text, new_text_rect)
            SCREEN.blit(matrix_maze_text, matrix_maze_text_rect)

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
                    print()
                    if(l[0]*l[1]==len(l[2])):
                        show_user_input(int(l[0]),int(l[1]),l[2])
                    else:
                        Tk().wm_withdraw() #to hide the main window
                        messagebox.showinfo('MessagePop','ERROR: Enter the parameters Again!!!')
                        get_user_input()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        mainmenu(username)

                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

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

def open_file():# import webbrowser
    if platform.system()=="Windows":
        osCommandString = "notepad.exe .configure.txt"
        os.system(osCommandString)
    elif platform.system()=="Linux":
        os.system("subl ./.configure.txt")
    else:
        Tk().wm_withdraw() #to hide the main window
        messagebox.showinfo('SORRY','Unknown OS detected, please open the file .configure.txt in any text editor and click the reload button')
                #webbrowser.open("configure.txt")
def camera_input():
    cam(900,600,int(600/25),1,0,1)

def display(font,font_size,text,pox,poy):
    if(text=="ABOUT PAGE" or text=="CONFIGURATION"):
        new_text = pygame.font.SysFont(f"{font}",font_size,bold=True).render(f"{text}", True, (252, 248, 232))
    else:
        new_text = pygame.font.SysFont(f"{font}",font_size).render(f"{text}", True, (252, 248, 232))
    new_text_rect = new_text.get_rect(center=(pox,poy))
    SCREEN.blit(new_text, new_text_rect)

def repolink(link,repo,pox,poy,size):
    link_font = pygame.font.SysFont('gabriola', size)
    link_color = (255, 255, 255)
    rect = SCREEN.blit(link_font.render(f"{repo}", True, link_color), (pox, poy))

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            if rect.collidepoint(pos):
                webbrowser.open(rf"{link}")

            if rect.collidepoint(pygame.mouse.get_pos()):
                link_color = (70, 29, 219)

        else:
            link_color = (0, 0, 0)        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

def about_info():
    WIDTH, HEIGHT = 1000, 600
    size=HEIGHT/25
    font = pygame.font.SysFont(None, int(size))
    pygame.init()
    
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ABOUT PAGE | MAZE SOLVER")
    clock = pygame.time.Clock()
    
    msgcol=(200,0,0)
    conretcolor=(0,0,0)
    conretclick=(255,255,255)
    msgcon = font.render("Open .configure.txt", 1, conretcolor)
    msgret = font.render('Reload', True, conretcolor)
    conrect = msgcon.get_rect()
    retrect = msgret.get_rect()
    conrect.center = (1.5*WIDTH//6,9*HEIGHT/10)
    retrect.center = (4.5*WIDTH//6,9*HEIGHT/10)
    msgconclk = font.render("Open .configure.txt", 1, conretclick)
    msgretclk = font.render('Reload',True, conretclick)
    conrectclk = msgconclk.get_rect()
    retrectclk = msgretclk.get_rect()
    conrectclk.center = (1.5*WIDTH//6,9*HEIGHT/10)
    retrectclk.center = (4.5*WIDTH//6,9*HEIGHT/10)
    butboxcol=(0,0,0)
    butboxclk=(255,255,255)
    conbox=pygame.Rect(0,0,(HEIGHT/20)*8,(HEIGHT/20))
    retbox=pygame.Rect(0,0,(HEIGHT/20)*8,(HEIGHT/20))
    conbox.center=conrect.center
    retbox.center=retrect.center
    while True:
           
            def button():
                pass

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            backbutton = pygame.image.load("back.png")
            backbutton = pygame.transform.scale(backbutton,(WIDTH/16,HEIGHT/12))
            biconc=backbutton.copy()
            chcolor(biconc,(231, 171, 121))
            brect=backbutton.get_rect()
            brectc=biconc.get_rect()
            SCREEN.blit(bg_img,(0,0))
            SCREEN.blit(backbutton,(0,0)) # paint to screen
            # pygame.display.flip()
            if is_over(brect,pygame.mouse.get_pos()):
                surface.blit(backbutton,brect)
            else:
                surface.blit(biconc,brectc)

          
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                elif ev.type==pygame.MOUSEBUTTONDOWN and (is_over(retbox,pygame.mouse.get_pos())):
                    # flag-=1
                    configure()
                    break
                elif ev.type==pygame.MOUSEBUTTONDOWN and is_over(conbox,pygame.mouse.get_pos()):
                    print("inin")
                    open_file()
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    x, y = ev.pos
                    if backbutton.get_rect().collidepoint(x, y):
                        return
            

            display("gabriola",WIDTH//32,"ABOUT PAGE",WIDTH/2,HEIGHT/13)
            display("gabriola",WIDTH//40,"We have used the modified DFS and A* to solve the MAZE",WIDTH/3,(HEIGHT+(HEIGHT/4))/7.5)
            display("gabriola",WIDTH//40,"It can take input from various sources i.e.",WIDTH/3.7,(HEIGHT+((1.5*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//40,"Web Input, File Input, Text Input, User Input",WIDTH/2.9,(HEIGHT+((2.5*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//40,"Modules that we have used are: ",WIDTH/4.2,(HEIGHT+((3.5*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//40,"pygame, tkinter, opencv, numpy, pillow",WIDTH/3.15,(HEIGHT+((4.5*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//40,"MADE BY and Source Code Link",WIDTH/4,(HEIGHT+((5.5*HEIGHT)/2))/7.5)
            repolink("https://github.com/rocklouis055/Maze","LOUISE PATRA: Repo 1  (click to follow)",WIDTH/5.3,(HEIGHT+((6.3*HEIGHT)/2))/7.5,WIDTH//32)
            repolink("https://github.com/Technocharm","MANISHA KUMARI: Repo 2  (click to follow)",WIDTH/5.3,(HEIGHT+((7*HEIGHT)/2))/7.5,WIDTH//32)
            repolink("./","MEHUL SINGH",WIDTH/5.3,(HEIGHT+((8.3*HEIGHT)/2))/7.5,WIDTH//32)
            # display("gabriola",WIDTH//40,"MEHUL SINGH",WIDTH/5.3,(HEIGHT+((8.3*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//32,"CONFIGURATION",WIDTH/2,(HEIGHT+((9.4*HEIGHT)/2))/7.5)
            display("gabriola",WIDTH//40,"User can configure the data of font, color, camera driver, user name and more",WIDTH/2.46,(HEIGHT+((10.5*HEIGHT)/2))/7.5)
            # manager = pygame_gui.UIManager((1000, 600))
            # text_input = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(((WIDTH-200)/2, ((HEIGHT-50)/2)), (550, 50)),text=".configure_file",manager=manager,object_id='#matrix_maze',visible=1)
            clock.tick(60)
            button()
            if is_over(conbox,pygame.mouse.get_pos()):
                SCREEN.blit(msgconclk,conrectclk)
                pygame.draw.rect(SCREEN,butboxclk,conbox,3,7)
            else:
                SCREEN.blit(msgcon,conrect)
                pygame.draw.rect(SCREEN,butboxcol,conbox,3,7)
            if is_over(retbox,pygame.mouse.get_pos()):
                SCREEN.blit(msgretclk,retrectclk)
                pygame.draw.rect(SCREEN,butboxclk,retbox,3,7)
            else:
                SCREEN.blit(msgret,retrect)
                pygame.draw.rect(SCREEN,butboxcol,retbox,3,7)
            pygame.display.update()


"""#########   _MAIN_ PROGRAM STARTS HERE    #########"""
def mainmenu(username):
    default=' Louise :) ' 
    # menu = pygame_menu.Menu(f'Welcome {default}', 1300, 700,
    #                        theme=pygame_menu.themes.THEME_BLUE)
    # font=pygame_menu.font.FONT_HELVETICA

    mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0),widget_font=('gabriola'), # transparent background    
                widget_font_color=(0, 0, 0)
            )
    mytheme.title_background_color=(191, 0, 255)
    mytheme.title_font_color=(255,255,255)



    myimage = pygame_menu.baseimage.BaseImage(  
    './img9.jpg',drawing_mode=pygame_menu.baseimage.IMAGE_MODE_FILL,
    drawing_offset=(0,0)
    )
    mytheme.background_color = myimage
    menu = pygame_menu.Menu(f'Welcome {username}', 1000, 600,
                        theme=mytheme )

    # menu.add.text_input('Enter Your Name :', default)
    # menu.add.selector('Mode :', [('Computer', 1), ('User', 2)], onchange=set_mode)
    menu.add.button('> Auto Generator and Solver',genmaze_solve)
    menu.add.button('> Play the Maze Game', start_the_game)
    menu.add.button('> Image from WEB',web_input)
    menu.add.button('> File input',file_input)
    menu.add.button('> Text input',text_input)
    menu.add.button('> Camera input',camera_input)
    menu.add.button('> About',about_info)
    menu.add.button('> Quit', pygame_menu.events.EXIT)

    menu.mainloop(surface)

# mainmenu()

def welcome_screen():
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("WELCOME | WINDOW")

    manager = pygame_gui.UIManager((1000, 600))
    
    text_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(((WIDTH+600)/4, HEIGHT/1.85), (200, 50)), manager=manager,object_id='#user_name')
    
    clock = pygame.time.Clock()

    def show_user_input(username):
        print("in")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            SCREEN.blit(bg_img,(0,0))

            mainmenu(username)

            clock.tick(60)

            pygame.display.update()

    def get_user_input():
        while True:
            UI_REFRESH_RATE = clock.tick(60)/1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and
                    event.ui_object_id == '#user_name'):
                        username=event.text
                        show_user_input(username)

                manager.process_events(event)
        
            manager.update(UI_REFRESH_RATE)

            # SCREEN.fill("black")
            bg_img = pygame.image.load('Img9.jpg')
            bg_img = pygame.transform.scale(bg_img,(WIDTH,HEIGHT))
            SCREEN.blit(bg_img,(0,0))

            manager.draw_ui(SCREEN)
            wel_text = pygame.font.SysFont("gabriola", WIDTH//8).render("THE MAZE", True, "white")
            wel_text_rect = wel_text.get_rect(center=(WIDTH/2, HEIGHT/4))
            SCREEN.blit(wel_text, wel_text_rect)

            new_text = pygame.font.SysFont("gabriola", WIDTH//16).render("Enter Your Name:", True, "white")
            new_text_rect = new_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            SCREEN.blit(new_text, new_text_rect)

            con_text = pygame.font.SysFont("gabriola", WIDTH//24).render("Press Enter to Continue", True, "white")
            con_text_rect = con_text.get_rect(center=(WIDTH/2, HEIGHT/1.5))
            SCREEN.blit(con_text, con_text_rect)

            clock.tick(60)
            pygame.display.update()
    get_user_input()

welcome_screen()



# game(50,50,720,720,50*50/2)
# auto(50,50,720,720,50*50/2)
# m,n=20,10
# k,path,t=genMaze(m=m,n=n)
# mazeanim(m,n,k,720,720,n/25,m*n/2)



