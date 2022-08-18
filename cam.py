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
cam(900,600,int(600/25),1,0,1)
