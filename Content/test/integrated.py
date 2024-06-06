from regularizer import regualarizer, window_regualarizer, xml_window_write
from models import bicycle_gan_model
from options.test_options import TestOptions
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import torch
import cv2
import xml.etree.ElementTree as ET

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, probability=0.6):
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)  # Ensure the tensor values are within [0, 1]
        return tensor

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, noise=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if noise:
            transform_list += [AddGaussianNoise(mean=0.0, std=0.1, probability=0.5)]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

opt = TestOptions().parse()

opt.num_threads = 1 
opt.batch_size = 1
opt.serial_batches = True

# create model
model = bicycle_gan_model.BiCycleGANModel(opt)
print("model [%s] was created" % type(model).__name__)

def xml_rewrite(xmlpath,box):
    y=box[0][3]-box[0][2]
    x=box[0][1]-box[0][0]
    tree=ET.parse(xmlpath)
    root=tree.getroot()
    for window in root.findall("window"):
        bndbox=window.find("bndbox")
        bndbox.find("xmin").text=str(int((int(bndbox.find("xmin").text)-box[0][0])*255/x))
        bndbox.find("xmax").text=str(int((int(bndbox.find("xmax").text)-box[0][0])*255/x))
        bndbox.find("ymin").text=str(int((int(bndbox.find("ymin").text)-box[0][2])*255/y))
        bndbox.find("ymax").text=str(int((int(bndbox.find("ymax").text)-box[0][2])*255/y))
    for window in root.findall("door"):
        bndbox=window.find("bndbox")
        bndbox.find("xmin").text=str(int((int(bndbox.find("xmin").text)-box[0][0])*255/x))
        bndbox.find("xmax").text=str(int((int(bndbox.find("xmax").text)-box[0][0])*255/x))
        bndbox.find("ymin").text=str(int((int(bndbox.find("ymin").text)-box[0][2])*255/y))
        bndbox.find("ymax").text=str(int((int(bndbox.find("ymax").text)-box[0][2])*255/y))
    for window in root.findall("window_pane"):
        bndbox=window.find("bndbox")
        bndbox.find("xmin").text=str(int((int(bndbox.find("xmin").text)-box[0][0])*255/x))
        bndbox.find("xmax").text=str(int((int(bndbox.find("xmax").text)-box[0][0])*255/x))
        bndbox.find("ymin").text=str(int((int(bndbox.find("ymin").text)-box[0][2])*255/y))
        bndbox.find("ymax").text=str(int((int(bndbox.find("ymax").text)-box[0][2])*255/y))
        
    tree.write(xmlpath, encoding='utf-8', xml_declaration=True)
    
def set_up_model(network):
    model.setup(opt,network)
    model.eval()
    print('Loading model %s' % opt.model)
    assert(opt.load_size >= opt.crop_size)

    if opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

def create_facade_input(size,offset):
    result= np.zeros((256, 256, 3), np.uint8)
    scale=(255-30)/max(size[0],size[1])
    size=(int(scale*size[0]),int(scale*size[1]))
    box=np.array([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]])
    box+=[int((255-size[0])/2),int((255-size[1])/2)]
    c=(0,0,255)
    cv2.fillPoly(result,box,c)
            
    cv2.imwrite("{}I.png".format(offset), result[:, :, [2, 1, 0]])

def create_img(inputpath,stylepath=None,noise=False):
    # test stage
    A_path = inputpath
    if stylepath is None:
        B_path = inputpath
    else:
        B_path = stylepath

    A = Image.open(A_path).convert('RGB')
    B = Image.open(B_path).convert('RGB')

    # apply the same transform to both A and B
    transform_params = get_params(opt, (256,256))
    A_transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1),noise=noise)
    B_transform = get_transform(opt, transform_params, grayscale=(opt.output_nc == 1),noise=noise)

    A = torch.reshape(A_transform(A), (1, 3, 256, 256))
    B = torch.reshape(B_transform(B), (1, 3, 256, 256))

    data={'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    model.set_input(data)
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)
            
    if stylepath is None:
        image_tensor = images[4].data
    else:
        image_tensor = images[2].data
        
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    result=image_numpy.astype(np.uint8)
    
    return result

def create_facade(mask,offset,style=None):
    box=create_facde_labemap(mask,offset)
    box.sort(key=lambda x:(x[3]-x[2])*(x[1]-x[0]),reverse=True)
    print(box)
    create_facade_texture("{}L.png".format(offset),offset,style)
    crop_facade(offset,box)
    xml_rewrite("{}.xml".format(offset),box)

def create_facde_labemap(mask,offset):
    set_up_model("./network/mask2coarselabelmap")
        
    result=create_img(mask)
    
    aligned_img, box=regualarizer(result,offset)

    cv2.imwrite("{}L.png".format(offset), aligned_img[:, :, [2, 1, 0]])
    return box
    
def create_facade_texture(labelmap,offset,style):
    set_up_model("./network/coarselabelmap2realimg")
    if style:
        result=create_img(labelmap,style)
    else:    
        result=create_img(labelmap)

    cv2.imwrite("{}.png".format(offset), result[:, :, [2, 1, 0]])

def crop_facade(offset,box):
    img=cv2.imread("{}.png".format(offset))[:, :, [2, 1, 0]]
    cv2.imwrite("{}.png".format(offset),img[
        box[0][2]:box[0][3], 
        box[0][0]:box[0][1], 
        [2, 1, 0]
        ])
    
    img=cv2.imread("{}L.png".format(offset))[:, :, [2, 1, 0]]
    cv2.imwrite("{}L.png".format(offset),img[
        box[0][2]:box[0][3], 
        box[0][0]:box[0][1], 
        [2, 1, 0]
        ])
    
def creat_windows(xmlpath,offset,style=None):
    tree=ET.parse(xmlpath)
    root=tree.getroot()
    done=set()
    for window in root.findall("window"):
        id=int(window.find("ID").text)
        if id in done:
            continue
        done.add(id)
        size=(int(window.find("size").find("width").text),int(window.find("size").find("height").text))
        creat_window(size,id,offset,style)

def creat_window(size,id,offset,style):
    create_window_input(size,id,offset)
    box=create_window_labelmap("{}I.png".format(id+offset+2),id,offset)
    create_window_texture("{}L.png".format(id+offset+2),id,offset,style)
    crop_window(id,offset,box)
    xml_rewrite("{}.xml".format(id+offset+2),box)

def create_window_input(size,id,offset):
    result= np.zeros((256, 256, 3), np.uint8)
    scale=(255-10)/max(size[0],size[1])
    size=(int(scale*size[0]),int(scale*size[1]))
    box=np.array([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]])
    box+=[int((255-size[0])/2),int((255-size[1])/2)]
    c=(255,0,0)
    cv2.fillPoly(result,box,c)
            
    cv2.imwrite("{}I.png".format(id+offset+2), result[:, :, [2, 1, 0]])

def create_window_labelmap(mask,id,offset):
    set_up_model("./network/window_mask2window_labelmap")
    result=create_img(mask)
    
    aligned_img, box=window_regualarizer(result,id,offset)
    
    cv2.imwrite("{}L.png".format(id+offset+2), aligned_img[:, :, [2, 1, 0]])
    
    return box
    
def create_window_texture(labelmap,id,offset,style=None):
    set_up_model("./network/window_labelmap2window_realimg")
    if style:
        result=create_img(labelmap,style)
    else:
        result=create_img(labelmap)
    cv2.imwrite("{}.png".format(id+offset+2), result[:, :, [2, 1, 0]])

def crop_window(id,offset,box):
    img=cv2.imread("{}.png".format(id+offset+2))[:, :, [2, 1, 0]]
    cv2.imwrite("{}.png".format(id+offset+2),img[
        box[0][2]:box[0][3], 
        box[0][0]:box[0][1], 
        [2, 1, 0]
        ])
    

width,depth,height=24,20,20
for i in range(4):
    if i%2:
        temp=depth
    else:
        temp=width
    create_facade_input((temp,height),16*i)
    create_facade("{}I.png".format(16*i),16*i,"facade_style4.jpg")
    creat_windows("{}.xml".format(16*i),16*i,"style5.png")