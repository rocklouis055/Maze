import numpy as np
import matplotlib.pyplot as plt
import cv2
def tup(p):
    return (int(p[0]), int(p[1]));

# load image
img = cv2.imread("maze.png");

# resize
scale = 0.5;
h, w = img.shape[:2];
h = int(h*scale);
w = int(w*scale);
img = cv2.resize(img, (w,h));
copy = np.copy(img);

# mask image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
mask = cv2.inRange(gray, 100, 255);

# find corners
corners = [[[0,0], 0] for a in range(4)];
for y in range(h+1):
    # progress check
    print(str(y) + " of " + str(h));
    for x in range(w):
        # check pixel
        if mask[y][x] == 0:
            # scores
            scores = [];
            scores.append((h - y) + (w - x)); # top-left
            scores.append((h - y) + x); # top-right
            scores.append(y + x); # bottom-right
            scores.append(y + (w - x)); # bottom-left
            
            # check corners
            for a in range(len(scores)):
                if scores[a] > corners[a][1]:
                    corners[a][1] = scores[a];
                    corners[a][0] = [x, y];

# draw connecting lines
for a in range(len(corners)):
    prev = corners[a-1][0];
    curr = corners[a][0];
    cv2.line(img, tup(prev), tup(curr), (0,200,0), 2);

# draw corners
for corner in corners:
    cv2.circle(img, tup(corner[0]), 4, (255,255,0), -1);

# re-orient to make the math easier
rectify = np.array([[0,0], [w,0], [w,h], [0,h]]);
numped_corners = [corner[0] for corner in corners];
numped_corners = np.array(numped_corners);
hmat, _ = cv2.findHomography(numped_corners, rectify);
rect = cv2.warpPerspective(copy, hmat, (w,h));

# redo mask
gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY);
mask = cv2.inRange(gray, 100, 255); 

# dilate
kernel = np.ones((3,3), np.uint8);
mask = cv2.erode(mask, kernel, iterations = 5);

# find entrances
top = []; # [score, point]
# top side
for x in range(w):
    y = 0;
    while mask[y][x] == 255:
        y += 1;
    top.append([y, [x,0]]);
# left side
left = [];
for y in range(h):
    x = 0;
    while mask[y][x] == 255:
        x += 1;
    left.append([x, [0,y]]);
# bottom side
bottom = [];
for x in range(w):
    y = h-1;
    while mask[y][x] == 255:
        y -= 1;
    bottom.append([(h - y) - 1, [x, h-1]]);
# right side
right = [];
for y in range(h):
    x = w-1;
    while mask[y][x] == 255:
        x -= 1;
    right.append([(w - x) - 1, [w-1,y]]);

# combine
scores = [top, left, bottom, right];

# plot
# for side in scores:
#     print("con 1")
#     fig = plt.figure();
#     print("con 1")
#     ax = plt.axes();
#     print("con 1")
#     y = [score[0] for score in side];
#     x = [a for a in range(len(y))];
#     ax.plot(x, y);
    
#     plt.show();

# get the top score for each side
highscores = []; # [score, [x, y]];
for side in scores:
    top_score = -1;
    top_point = [-1, -1];
    for score in side:
        if score[0] > top_score:
            top_score = score[0];
            top_point = score[1];
    highscores.append([top_score, top_point]);

# get the top two (assuming that there are two entrances to the maze)
one = [0, [0,0]];
two = [0, [0,0]];
for side in highscores:
    if side[0] > one[0]:
        two = one[:];
        one = side[:];
    elif side[0] > two[0]:
        two = side[:];

# draw the entrances
cv2.circle(rect, tup(one[1]), 5, (0,0,255), -1);
cv2.circle(rect, tup(two[1]), 5, (0,0,255), -1);

# show
cv2.imshow("Image", img);
cv2.imshow("Rect", rect);
cv2.waitKey(0);