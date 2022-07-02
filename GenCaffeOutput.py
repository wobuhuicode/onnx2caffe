#from __future__ import print_function 
import caffe
from datetime import datetime
import numpy as np
import struct
import sys, getopt
import cv2, os
import pickle as p
import matplotlib.pyplot as pyplot
import ctypes
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as caffe_protobuf
import platform
import argparse
from tqdm import tqdm
import google.protobuf.text_format

cpu_supported_layers=[
    "Convolution", "Deconvolution", "Pooling", "InnerProduct", 
    "LRN", "BatchNorm", "Scale", "Bias", 
    "Eltwise", "ReLU", "PReLU", "AbsVal", 
    "TanH", "Sigmoid", "BNLL", "ELU", 
    "LSTM", "RNN", "Softmax", "Exp", 
    "Log", "Reshape", "Flatten", "Split", 
    "Slice", "Concat", "SPP", "Power", 
    "Threshold", "MVN", "Parameter", "Reduction", 
    "Input", "Dropout", "ROIPooling", "Upsample", 
    "Normalize", "Permute", "PSROIPooling", "PassThrough", 
    "Python"]

cuda_supported_layers=[
    "Convolution", "Deconvolution", "Pooling", "InnerProduct", 
    "LRN", "BatchNorm", "Scale", "Bias", 
    "Eltwise", "ReLU", "PReLU", 
    "AbsVal", "TanH", "Sigmoid", "BNLL", 
    "ELU", "LSTM", "RNN", "Softmax", "Exp", 
    "Log", "Reshape", "Flatten", "Split", 
    "Slice", "Concat", "SPP", "Power", 
    "Threshold", "MVN", "Parameter", "Reduction", 
    "Input", "Dropout"]

def isValidNormType(normType):
    if (normType == '0' or normType == '1' or \
        normType == '2' or normType == '3' or \
        normType == '4' or normType == '5') :
        return True
    return False

def isSupportedLayer(layer_type, cuda_flag):
    if '1' == cuda_flag:
        for type in cuda_supported_layers:
            if(layer_type == type):
                return True
    else:
        for type in cpu_supported_layers:
            if(layer_type == type):
                return True
    return False

def judge_supported_layer(model_filename, train_net, cuda_flag):
    if(platform.system()=="Linux"):
        f=open(model_filename, 'rb')
    else:
        f=open(model_filename.encode('gbk'), 'rb')

    train_str = f.read()
    
    caffe_protobuf.text_format.Parse(train_str, train_net)
    f.close()
    layers = train_net.layer
 
    for layer in layers:
        if(False == isSupportedLayer(layer.type, cuda_flag)):
            print("Layer " + layer.name + " with type " + layer.type + \
            " is not supported, please refer to chapter 3.1.4 and FAQ  \
            of \"HiSVP Development Guide.pdf\" to extend caffe!")
            
            sys.exit(1)

def print_log_info(model_filename, weight_filename, cfg, output_dir):
    print('model file is: ' + model_filename)
    print('weight file is: ' + weight_filename)
    print('output dir is: ' + output_dir)
    for i in range(len(cfg.image_file)):
        print('data number: ' + str(i))
        print('image file is: ' + cfg.image_file[i])
        print('image preprocessing method is: ' + str(cfg.norm_type[i]))
        print('data_scale is: ' + str(cfg.data_scale))

def isfloat(value):
    try:
       float(value)
       return True
    except ValueError:
       return False

def isValidDataScale(value):
    if(True == isfloat(value)):
        if(float(value) >= 0.000244140625 and float(value) <= 4294967296.0):
           return True
        else:
            return False
    else:
        return False

def image_to_array(img_file, shape_c_h_w, output_dir):
    result = np.array([])
    print("converting begins ...")
    resizeimage = cv2.resize(cv2.imread(img_file), (shape_c_h_w[2],shape_c_h_w[1]))
    b,g,r = cv2.split(resizeimage)
    height, width, channels = resizeimage.shape
    length = height*width
    #print(channels )
    r_arr = np.array(r).reshape(length)
    g_arr = np.array(g).reshape(length)
    b_arr = np.array(b).reshape(length)
    image_arr = np.concatenate((r_arr, g_arr, b_arr))
    result = image_arr.reshape((1, length*3))
    print("converting finished ...")
    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.bin"%(channels,height,width))
    with open(file_path, mode='wb') as f:
         p.dump(result, f)
    print("save bin file success")

def image_to_rgb(img_file, shape_c_h_w, output_dir):
    print("converting begins ...")

    image = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), 1)
    image = cv2.resize(image, (shape_c_h_w[2],shape_c_h_w[1]))
    image = image.astype('uint8')
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.rgb"%(channels,height,width))
    fileSave =  open(file_path,'wb')

    for step in range(0,height):
        for step2 in range (0, width):
            fileSave.write(image[step,step2,2])
    for step in range(0,height):
        for step2 in range (0, width):
            fileSave.write(image[step,step2,1])
    for step in range(0,height):
        for step2 in range (0, width):
            fileSave.write(image[step,step2,0])

    fileSave.close()
    print("converting finished ...")

def image_to_bin(img_file, shape_c_h_w, output_dir):
    print("converting begins ...")

    image = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    image = image.astype('uint8')
    height = image.shape[0]
    width = image.shape[1]

    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.bin"%(1,height,width))
    fileSave =  open(file_path,'wb')
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(image[step,step2])

    fileSave.close()
    print("converting finished ...")

def image_to_bgr(img_file,shape_c_h_w, output_dir):
    print("converting begins ...")
    image = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
    image = cv2.resize(image, (shape_c_h_w[2],shape_c_h_w[1]))
    image = image.astype('uint8')
    b,g,r = cv2.split(image)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.bgr"%(channels,height,width))
    fileSave =  open(file_path,'wb')
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(b[step,step2])
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(g[step,step2])
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(r[step,step2])

    fileSave.close()
    print("converting finished ...")

def bin_to_image(bin_file,shape_c_h_w):
    if(platform.system()=="Linux"):
        fileReader = open(bin_file,'rb')
    else:
        fileReader = open(bin_file.encode('gbk'),'rb')

    height = shape_c_h_w[1]
    width = shape_c_h_w[2]
    channel = shape_c_h_w[0]
    imageRead = np.zeros((shape_c_h_w[1], shape_c_h_w[2], shape_c_h_w[0]), np.uint8)

    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,2] = a[0]
    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,1] = a[0]
    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,0] = a[0]
    fileReader.close()

    return imageRead

def rgb_to_image(bin_file,shape_c_h_w):
    if(platform.system()=="Linux"):
        fileReader = open(bin_file,'rb')
    else:
        fileReader = open(bin_file.encode('gbk'),'rb')

    height = shape_c_h_w[1]
    width = shape_c_h_w[2]
    channel = shape_c_h_w[0]
    imageRead = np.zeros((shape_c_h_w[1], shape_c_h_w[2], shape_c_h_w[0]), np.uint8)

    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,2] = a[0]
    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,1] = a[0]
    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,0] = a[0]
    fileReader.close()

    return imageRead

def bgr_to_image(bin_file, shape_c_h_w):
    if(platform.system()=="Linux"):
        fileReader = open(bin_file,'rb')
    else:
        fileReader = open(bin_file.encode('gbk'),'rb')

    height = shape_c_h_w[1]
    width = shape_c_h_w[2]
    channel = shape_c_h_w[0]
    imageRead = np.zeros((shape_c_h_w[1], shape_c_h_w[2], shape_c_h_w[0]), np.uint8)
    
    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,0] = a[0] 
    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,1] = a[0]
    for step in range(0,height):
        for step2 in range (0, width):
            a = struct.unpack("B", fileReader.read(1))
            imageRead[step,step2,2] = a[0]
    fileReader.close()

    return imageRead

def get_float_numbers(floatfile):
    mat = []

    if(platform.system()=="Linux"):
        with open(floatfile, 'rb') as input_file:
            for line in input_file:
                line = line.strip()
                for number in line.split():
                    if isfloat(number):
                        mat.append(float(number))
    else:
        with open(floatfile.encode('gbk'), 'rb') as input_file:
            for line in input_file:
                line = line.strip()
                for number in line.split():
                    if isfloat(number):
                        mat.append(float(number))
    return  mat

def isHex(value):
    try:
       int(value,16)
       return True
    except ValueError:
       return False

def isHex_old(value):
    strvalue=str(value)
    length = len(strvalue)
    if length == 0:
        return False

    i = 0
    while(i < length):
        if not (strvalue[i] >= 'a' and strvalue[i] <= 'e' or \
                strvalue[i] >= 'A' and strvalue[i] <= 'E' or \
                strvalue[i] >= '0' and strvalue[i] <= '9'):

            return False
        i += 1
    return True

def get_hex_numbers(hexfile):
    mat = []
    if(platform.system()=="Linux"):
        with open(hexfile) as input_file:
            for line in input_file:
                line = line.strip()
                for number in line.split():
                    if isHex(number):
                        mat.append(1.0*ctypes.c_int32(int(number,16)).value/4096)
    else:
       with open(hexfile.encode("gbk")) as input_file:
            for line in input_file:
                line = line.strip()
                for number in line.split():
                    if isHex(number):
                        mat.append(1.0*ctypes.c_int32(int(number,16)).value/4096)
    return mat 

# save result by layer name
def save_result(train_net, net, output_dir):
    max_len = len(train_net.layer)

    # input data layer
    index = 0
    for inputs in train_net.input:
        layer_data = net.blobs[inputs].data[...]
        layer_name=inputs.replace("/", "_").replace("-","_")
        shape_str= str(layer_data.shape)
        shape_str=shape_str[shape_str.find(", ") + 1:].replace("(", "").replace(")", "").replace(" ", "").replace(",", "_");
        filename = os.path.join(output_dir, "%s_output%d_%s_caffe.linear.float"%(layer_name, index, shape_str))
        np.savetxt(filename, layer_data.reshape(-1, 1))
        index = index + 1

    # other layer
    i = 0
    for layer in tqdm(train_net.layer):
        index = 0
        for top in layer.top:
            # ignore inplace layer
            if 1 == len(layer.top) and 1 == len(layer.bottom) and layer.top[0] == layer.bottom[0]:
                break
            layer_data = net.blobs[top].data[...]
            layer_name=layer.name.replace("/", "_").replace("-","_")
            shape_str= str(layer_data.shape)
            shape_str=shape_str[shape_str.find(", ") + 1:].replace("(", "").replace(")", "").replace(" ", "").replace(",", "_")
            filename = os.path.join(output_dir, "%s_output%d_%s_caffe.linear.float"%(layer_name, index, shape_str))
            np.savetxt(filename, layer_data.reshape(-1, 1))
            index = index + 1

class CfgParser:
    def __init__(self, cfg_path):

        fread = open(cfg_path, "r")
        cfg_data_lines = fread.readlines()
        self.image_file = []
        self.mean_file = []
        self.norm_type = []
        self.data_scale = []
  
        for line in cfg_data_lines:
            if( line.strip() == ""):
                continue
            if "[image_file]" == line.split()[0].strip():
                self.image_file.append(line.split()[1].strip())
            if "[mean_file]" == line.split()[0].strip():
                self.mean_file.append(line.split()[1].strip())
            if "[norm_type]" == line.split()[0].strip():
                if(False == isValidNormType(line.split()[1].strip())):
                    print("Error: Input parameter normType is not valid, \
                            range is 0 to 5 and it should be integer")
                    sys.exit(2)
                self.norm_type.append(line.split()[1].strip())
            if "[data_scale]" == line.split()[0].strip():
                if(True == isValidDataScale(line.split()[1].strip())):
                    self.data_scale = float(line.split()[1].strip())
                else:
                    self.data_scale = -1

def load_data_from_text(image_file, shape, net):
    if(image_file.endswith('.float')):
        data = np.asarray(get_float_numbers(image_file))
        inputs = data
        inputs= np.reshape(inputs, net.blobs[list(net.blobs.keys())[0]].data.shape)
        return inputs
    elif(image_file.endswith('.hex')):
        data = np.asarray(get_hex_numbers(image_file))
        inputs = data
        inputs= np.reshape(inputs, net.blobs[list(net.blobs.keys())[0]].data.shape)
        return inputs

def load_data_from_image(data_layer, cfg, i, net, output_dir):
    #preprocess
    img_filename = cfg.image_file[i]
    norm_type = cfg.norm_type[i]
    meanfile = cfg.mean_file[i]
    data_scale = cfg.data_scale
    # norm_type
    # 0: no preprocess
    # 1: mean file
    # 2: channel mean file
    # 3: data scale
    # 4: mean file with data scale
    # 5: channel mean file with data scale
    
    if norm_type == '1' or norm_type == '4': 
        if not os.path.isfile(meanfile):
           print("Please give the mean image file path") 
           sys.exit(1)
        if meanfile.endswith('.binaryproto'):
            meanfileBlob = caffe.proto.caffe_pb2.BlobProto()

            if(platform.system()=="Linux"):
                meanfileData = open(meanfile, 'rb').read()
            else:
                meanfileData = open(meanfile.encode('gbk'), 'rb').read()
            meanfileBlob.ParseFromString(meanfileData)
            arr = np.array(caffe.io.blobproto_to_array(meanfileBlob))
            out = arr[0]
            np.save('transMean.npy', out)
            meanfile = 'transMean.npy'

    if(img_filename.endswith('.bgr')):
        inputs = bgr_to_image(img_filename, net.blobs[data_layer].data.shape[1:])
    elif(img_filename.endswith('.rgb')):
        inputs = rgb_to_image(img_filename, net.blobs[data_layer].data.shape[1:])
    else:
        '''
        # input: N C H W
        # check input channel
        if net.blobs[data_layer].data.shape[1]==1:
            # one channel, gray, no color
            # color = False
            image_to_bin(img_filename, net.blobs[data_layer].data.shape[1:], output_dir)
        elif net.blobs[data_layer].data.shape[1]==3:
            # three channel, rgb image
            # color = True
            # transform image
            image_to_rgb(img_filename, net.blobs[data_layer].data.shape[1:], output_dir)
            image_to_bgr(img_filename, net.blobs[data_layer].data.shape[1:], output_dir)
        ''' 
        img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = img

    transformer = caffe.io.Transformer({data_layer: net.blobs[data_layer].data.shape})
    # HWC ==> CHW
    if net.blobs[data_layer].data.shape[1]==3:
        transformer.set_transpose(data_layer, (2,0,1))

    # use mean file normlize image
    if norm_type == '1' or norm_type == '4' and os.path.isfile(meanfile): # (sub mean by meanfile):  
        if net.blobs[data_layer].data.shape[1]==3:
            transformer.set_mean(data_layer,np.load(meanfile).mean(1).mean(1))
        elif net.blobs[data_layer].data.shape[1]==1:
            tempMeanValue = np.load(meanfile).mean(1).mean(1)
            tempa = list(tempMeanValue)
            inputs = inputs - np.array(list(map(float, [tempa[0]])))
    elif norm_type == '2' or norm_type == '5':
        if net.blobs[data_layer].data.shape[1]==3:
            mean_value = np.loadtxt(meanfile)
            print('mean channel value: ', mean_value)
            if len(mean_value) != 3: 
                print("Please give the channel mean value in BGR order with 3 values, like 112,113,120") 
                sys.exit(1)
            if not isfloat(mean_value[0]) or not isfloat(mean_value[1]) or not isfloat(mean_value[2]): 
                print("Please give the channel mean value in BGR order") 
                sys.exit(1)
            else:
                transformer.set_mean(data_layer, mean_value)
        elif net.blobs[data_layer].data.shape[1]==1:
            with open(meanfile, 'r') as f:
                lmeanfile = f.read().splitlines()
                print(lmeanfile)

            if isfloat(lmeanfile[0]):  # (sub mean by channel)
                inputs = inputs - np.array(list(map(float, [lmeanfile[0]])))

    elif norm_type == '3':
        transformer.set_input_scale(data_layer, data_scale)
       #inputs = inputs * float(data_scale)
    
    data = inputs
    if norm_type == '4' or norm_type == '5':
       data = data * float(data_scale)

    # set im_info for RCNN net
    if 'im_info' in net.blobs:
        data_shape = net.blobs[net.inputs[0]].data.shape
        im_shape = data.shape
        #logging.debug("data shape:" + str(data_shape))
        #logging.debug("image shape:" + str(im_shape))
        im_scale_height = float(data_shape[2])/float(im_shape[0])
        im_scale_width = float(data_shape[3])/float(im_shape[1])
        #if math.fabs(im_scale_height - im_scale_width) > 0.1:
        #    logging.warning("im_scale_height[%f] is not equal to im_scale_width[%f].\nPlease reshape data input layer to (%d, %d) in prototxt, otherwise it may detect failed."%(im_scale_height, im_scale_width, im_shape[0], im_shape[1]))
        # im_scale = data(w,h) / image(w,h)
        im_scale = im_scale_height if im_scale_height > im_scale_width else im_scale_width
        im_info_data = np.array([[data_shape[2], data_shape[3], im_scale]], dtype=np.float32).reshape(net.blobs['im_info'].data.shape)
        np.set_printoptions(suppress=True)
        #logging.debug("im_info:" + str(im_info_data))
        net.blobs['im_info'].data[...] = im_info_data
    print(data.shape)
    #data = transformer.preprocess(data_layer, data)
    data = cv2.resize(data, (640,640))
    data = (data*0.0039062).transpose((2, 0, 1))
    
    print(data)
    print(data.shape)
    return data

def preprocess_data(cfg, net, output_dir, train_net):
    input_layers = []
    for layer in train_net.layer:
        if(layer.type == 'Input'):
            input_layers.append(layer)
      
    # load image
    for i in range(len(cfg.image_file)):
        if i < len(net.inputs):
            data_layer = net.inputs[i]
            if (cfg.image_file[i].endswith('.float') or cfg.image_file[i].endswith('.hex')):
                input_data = load_data_from_text(cfg.image_file[i], net.blobs[data_layer].data.shape, net) 
            else:
                input_data = load_data_from_image(data_layer, cfg, i, net, output_dir)
            # load data to net
            print(type(input_data))
            print(input_data.shape)
            input_data = input_data[::-1]
            print("----------------", input_data.shape)
            net.blobs[data_layer].data[...] = input_data
        else:
            j = 0
            for top in input_layers[j].top:
                if (cfg.image_file[i].endswith('.float') or cfg.image_file[i].endswith('.hex')):
                    input_data = load_data_from_text(cfg.image_file[i], net.blobs[top].data.shape, net) 
                else:
                    input_data = load_data_from_image(top, cfg, i, net, output_dir)
                 # load data to net
                net.blobs[top].data[...] = input_data
                j = j + 1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', type=str, default='./lpr.cfg', help='.cfg, store the information of input data')
    parser.add_argument('--model', type=str, default='./lprnetMbv3X3-40000-sim.prototxt', help='.prototxt, batch num should be 1')
    parser.add_argument('--weight', type=str, default='./lprnetMbv3X3-40000-sim.caffemodel', help='.caffemodel')
    parser.add_argument('--output', type=str, default="./lprout", help='optional, if not set, there will be a directory named output created in out')
    parser.add_argument('--device', type=int, default=0, help='1, gpu, 0 cpu')
    opt = parser.parse_args()
    return opt

def main():
    # prase option
    # opt = parse_opt()
    cfg_filename = ["test_images/bdd.cfg"]
    model_name = "ghost_relu3"
    weight_filename = f"weights/{model_name}.caffemodel"
    model_filename = f"weights/{model_name}.prototxt"
    output_dir = f"output/{model_name}"
    cuda_flag = 0

    if not os.path.exists(f'output/{model_name}'):
        os.mkdir(f'output/{model_name}')

    # check device
    if(cuda_flag == '1'):
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    print(cfg_filename)

    # parse image params
    cfg = CfgParser(cfg_filename[0])

    if(False == isValidDataScale(cfg.data_scale)):
        print("The datascale in cfg file is invalid, \
               it should be no less than 0.000244140625 and no more than 4294967296.0")
        return
        
    #parse prototxt
    train_net = caffe_pb2.NetParameter()

    #judge if the prototxt has unspporterd layers
    judge_supported_layer(model_filename, train_net, cuda_flag)

    #print debug info
    #print_log_info(model_filename, weight_filename, cfg, output_dir)
    
    #load caffe prototxt and caffe model
    if(platform.system()=="Linux"):
        net = caffe.Net(model_filename, weight_filename, caffe.TEST)
    else:
        net = caffe.Net(model_filename.encode('gbk'), weight_filename.encode('gbk'), caffe.TEST)
    print ('model load success')
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # read data from file and normlize 
    preprocess_data(cfg, net, output_dir, train_net)

    out = net.forward()
    save_result(train_net, net, output_dir)
    return 

if __name__=='__main__':
    main()
