import cv2
import numpy as np
import random

img = cv2.imread('test.png')
mask = np.zeros(img.shape[:2], dtype=np.uint8)
patch_len = 5


def draw(event, x, y, flags, param):
    global mask
    global img
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
        cv2.rectangle(img, (x, y), (x + 7, y + 7), (255, 255, 255), -1)
        cv2.rectangle(mask, (x, y), (x + 7, y + 7), 255, -1)


def getdis(A, B, p, q):
    dis = abs(A[p[0]:p[0] + patch_len, p[1]:p[1] + patch_len] - B[q[0]:q[0] + patch_len, q[1]:q[1] + patch_len]).sum()
    return dis


def fill(src, tar, fr, to):
    tar[to[0]][to[1]] = src[fr[0]][fr[1]]

def PatchMatch(img, mask, level, prev_match, max_level):
    cor = []
    w, h = mask.shape
    idx = np.zeros((w, h), dtype=np.int)
    tar = img * ((mask == 0)[:, :, None])
    match = np.zeros((w, h, 2), dtype=np.int)
    for i in range(w - patch_len):
        for j in range(h - patch_len):
            if mask[i][j] != 0:
                idx[i][j] = len(cor)
                cor.append([i, j])
                if level == 0:
                    match[i][j] = np.array([np.random.randint(0, w - patch_len), np.random.randint(0, h - patch_len)])
                else:
                    match[i][j] = np.array([prev_match[i // 2][j // 2]]) * 2
            else:
                match[i][j] = np.array([i, j])
    cor = np.array(cor)
    n = len(cor)
    value = np.zeros(n, dtype=np.int)
    movex = [0, -1, 1, 0]
    movey = [1, 0, 0, -1]
    order = range(n)
    for i in order:
        x, y = cor[i]
        value[i] = getdis(img, tar, match[x, y], cor[i])
        fill(img, tar, match[x, y], cor[i])
    for epoch in range(20):
        print(epoch)
        order = order[::-1]
        for i in order:
            x, y = cor[i]
            for d in range(4):
                if epoch % 2 == d % 2:
                    newdis = getdis(img, tar, match[x + movex[d], y + movey[d]], cor[i])
                    if newdis < value[i]:
                        value[i] = newdis
                        match[x, y] = match[x + movex[d], y + movey[d]]
                        fill(img, tar, match[x, y], [x, y])
            R = max(w, h)
            v0 = match[x, y]
            while R >= 1:
                R /= 2.0
                r = int(R)
                u = v0 + np.random.randint(-r, r + 1, 2)
                while u[0] < 0 or u[1] < 0 or u[0] + patch_len >= w or u[1] + patch_len >= h:
                    u = v0 + np.random.randint(-r, r + 1, 2)
                newdis = getdis(img, tar, u, cor[i])
                if newdis < value[i]:
                    value[i] = newdis
                    match[x, y] = u
                    fill(img, tar, match[x, y], [x, y])
    if level == max_level:
        tar = tar.astype(np.uint8)
        cv2.imshow("result.png", tar)
        cv2.waitKey()
    return match

cv2.namedWindow('draw')
cv2.setMouseCallback('draw', draw)
while True:
    cv2.imshow('draw', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyWindow('draw')
img = cv2.imread('test.png')
imgs = [img.astype(np.int)]
masks = [mask]
while img.shape[0] > 10 * patch_len and img.shape[1] > 10 * patch_len:
    mask = cv2.pyrDown(mask)
    img = cv2.pyrDown(img)
    imgs.append(img.astype(np.int))
    masks.append(mask)
imgs = imgs[::-1]
masks = masks[::-1]
prev_match = None
for i in range(len(imgs)):
    prev_match = PatchMatch(imgs[i], masks[i], i, prev_match, len(imgs) - 1)