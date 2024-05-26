import cv2
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

facade_classes = [
        ('other', [0, 0, 0], 0),  # black borders or sky (id 0)
        ('background', [0, 0, 170], 1),  # background (id 1)
        ('facade', [0, 0, 255], 2),  # facade (id 2)
        ('moulding', [255, 85, 0], 3),  # moulding (id 3)
        ('cornice', [0, 255, 255], 4),  # cornice (id 4)
        ('pillar', [255, 0, 0], 5),  # pillar (id 5)
        ('window', [0, 85, 255], 6),  # window (id 6)
        ('door', [0, 170, 255], 7),  # door (id 7)
        ('sill', [85, 255, 170], 8),  # sill (id 8)
        ('blind', [255, 255, 0], 9),  # blind (id 9)
        ('balcony', [170, 255, 85], 10),  # balcony (id 10)
        ('shop', [170, 0, 0], 11),  # shop (id 11)
        ('deco', [255, 170, 0], 12),  # deco (id 12)
    ]
window_classes = [
        ('other', [0, 0, 0], 0),  # black borders or sky (id 0)
        ('window', [255, 0, 0], 1),  # window (id 1)
        ('window_pane', [0, 0, 255], 2),  # window_pane (id 2)
    ]

depth= [
        0,  # black borders or sky (id 0)
        -2,  # background (id 1)
        0,  # facade (id 2)
        0,  # moulding (id 3)
        0,  # cornice (id 4)
        0,  # pillar (id 5)
        -1,  # window (id 6)
        -1,  # door (id 7)
        0,  # sill (id 8)
        0,  # blind (id 9)
        0,  # balcony (id 10)
        0,  # shop (id 11)
        0,  # deco (id 12)
    ]

def find_nearest_class(img,classes):
    temp=np.stack([c[1] for c in classes]).reshape(-1, 1, 1, 3)
    class_ind = np.power(img - temp, 2).sum(axis=3).argmin(axis=0)
    
    return class_ind

def get_mask(class_ind,id,kernal,merge=True):
    mask = (class_ind == id).astype(np.uint8)
    if merge:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
        

    return mask

def find_boxes(mask,id,boxes):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j, contour in enumerate(contours):
        if hierarchy[0, j, 3] != -1:
            continue

        bbox = np.min(contour[:, :, 0]), np.max(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        area = cv2.contourArea(contour)

        if  area < 20:
            continue

        boxes[id].append(tuple(int(bbc) for bbc in bbox))

def draw_boxes(rimg,boxes,classes,istesselation=False):
    for i, boxclass in enumerate(boxes):
        for box in boxclass:
            temp1=np.array([128,128,128])
            temp2=np.array([128,128,128])
            c=(temp1+depth[i]*temp2/8) if istesselation else classes[i][1]
            cv2.fillPoly(
                rimg,
                np.array([[[box[0], box[2]], [box[0], box[3]], [box[1], box[3]], [box[1], box[2]]]]),
                c)

def fit_boxes(img,classes,merge):
    rimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    boxes = [[] for _ in classes]
    
    class_ind=find_nearest_class(img,classes)
    
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    for _, _, id in classes:
        mask=get_mask(class_ind,id,kernal,merge)
        
        if mask.sum() < 20:
            continue
        
        find_boxes(mask,id,boxes)
    if len(boxes)>7:
        n=len(boxes[7])
        for i in range(n):
            a,b,c,d=boxes[7][i]
            boxes[7][i]=(a,b,c,img.shape[0])
    
    return boxes

def align_boxes(boxes_,kernal_size):
    def K(x,x0,h):
        u=(x-x0)/h
        c=1/np.sqrt(2*np.pi)
        return c*np.exp(-u**2/2)
    
    boxes=[[0,0,0,0] for _ in boxes_]
    for i,box0_ in enumerate(boxes_):
        for j in range(4):
            x0=box0_[j]
            for _ in range(50):
                temp1=0
                temp2=0
                for box_ in boxes_:
                    x=box_[j]
                    temp1+=K(x,x0,kernal_size)
                    temp2+=K(x,x0,kernal_size)*x
                x0=temp2/temp1
            boxes[i][j]=int(x0)
    return boxes

dstfolder='result'
srcfolder=''

def windowcluster(K,boxes):
    size=set()
    for xmin,xmax,ymin,ymax in boxes:
        x,y=xmax-xmin,ymax-ymin
        if x==0 or y==0:
            continue
        size.add((x,y))
    size=list(size)
    size=np.array(size)
    
    if size.shape[0]<K:
        result=KNN(size.shape[0],size)
    else:
        result=KNN(K,size)
    
    return result

def KNN(K,size):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(size)
    dict={}
    for i in range(size.shape[0]):
        dict[(size[i][0],size[i][1])]=kmeans.labels_[i]
    print(dict)
    return dict

def xml_write(boxes,class_,dict,offset):
    root=ET.Element("facade")
    for i in range(len(class_)):
        classname=class_[i]
        for xmin,xmax,ymin,ymax in boxes[i]:
            if (xmax-xmin==0) or (ymax-ymin==0):
                continue
            
            temp=ET.Element(classname)
            root.append(temp)
            subelement1=ET.SubElement(temp,"bndbox")
            ET.SubElement(subelement1,"xmin").text=str(xmin)
            ET.SubElement(subelement1,"ymin").text=str(ymin)
            ET.SubElement(subelement1,"xmax").text=str(xmax)
            ET.SubElement(subelement1,"ymax").text=str(ymax)
            
            subelement2=ET.SubElement(temp,"size")
            ET.SubElement(subelement2,"width").text=str(xmax-xmin)
            ET.SubElement(subelement2,"height").text=str(ymax-ymin)
            
            if i==0:
                ET.SubElement(temp,"ID").text=str(dict[(xmax-xmin,ymax-ymin)])
    
    tree=ET.ElementTree(root)
    
    fileName = "{}.xml".format(offset)
    with open(fileName, "wb") as file:
        tree.write(file, encoding='utf-8', xml_declaration=True)
        
def xml_window_write(box,id,offset):
    root=ET.Element("window")
    for xmin,xmax,ymin,ymax in box:
        if (xmax-xmin==0) or (ymax-ymin==0):
            continue
        
        temp=ET.Element("window_pane")
        root.append(temp)
        subelement1=ET.SubElement(temp,"bndbox")
        ET.SubElement(subelement1,"xmin").text=str(xmin)
        ET.SubElement(subelement1,"ymin").text=str(ymin)
        ET.SubElement(subelement1,"xmax").text=str(xmax)
        ET.SubElement(subelement1,"ymax").text=str(ymax)
        
        subelement2=ET.SubElement(temp,"size")
        ET.SubElement(subelement2,"width").text=str(xmax-xmin)
        ET.SubElement(subelement2,"height").text=str(ymax-ymin)

    tree=ET.ElementTree(root)
    
    fileName = "{}.xml".format(id+offset+2)
    with open(fileName, "wb") as file:
        tree.write(file, encoding='utf-8', xml_declaration=True)

def regualarizer(img,offset):
    boxes=fit_boxes(img,facade_classes,True)
    fitted_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(fitted_img,boxes,facade_classes)
    
    boxes[6]=align_boxes(boxes[6][:],6)
    boxes[7]=align_boxes(boxes[7][:],4)
    dict=windowcluster(4,boxes[6])
    xml_write([boxes[6],boxes[7]],["window","door"],dict,offset) 
    
    aligned_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(aligned_img,boxes,facade_classes)
    
    tesselation_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(tesselation_img,boxes,facade_classes,True)
    
    return aligned_img, boxes[1]

def window_regualarizer(img,id,offset):
    boxes=fit_boxes(img,window_classes,False)
    fitted_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(fitted_img,boxes,window_classes)
    
    boxes[2]=align_boxes(boxes[2][:],4)
    xml_window_write(boxes[2],id,offset) 
    
    aligned_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(aligned_img,boxes,window_classes)
    
    tesselation_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    draw_boxes(tesselation_img,boxes,window_classes,True)
    
    return aligned_img, boxes[1]

if __name__=="__main__":
    img=cv2.imread("0L.png")[:, :, [2, 1, 0]]
    regualarizer(img,64)