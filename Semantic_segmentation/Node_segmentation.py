#!/usr/bin/env python

import rospy
import numpy as np 
import torch
from PIL import Image as Img
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time

import glob


from torchvision.transforms import ToPILImage,ToTensor
from torch.autograd import Variable
from utils.Color import Colorize
from utils.erfnet import Net



rospy.init_node('roads_lanes_obstacles_segmentation') #nombre del nodo
pub=rospy.Publisher('/segmented_image',Image,queue_size=1) #nombre de lo que voy a publicar segmentado
pub_orig=rospy.Publisher('/image_orig',Image,queue_size=1) #nombre de lo que voy a publicar imagen




flag=False

#NUM_CLASSES=7
NUM_CLASSES=16
color_transform=Colorize()


contador=0
tiempo=0

#img_array = []

#size = (640,480)

##NUM_CLASSES
##CvBridge funcion de cvbridge
class resend:
    def __init__(self):
        global NUM_CLASSES
        self.bridge = CvBridge()

        #Inicializacion de la red
        self.model = Net(NUM_CLASSES)
        self.model = torch.nn.DataParallel(self.model).cuda()
        #Carga de pesos
        self.model.load_state_dict(torch.load("utils/model_best.pth"))
        print ("Model and weights LOADED successfully")
        self.model.eval()
## bga video del carrito
## como correr un vag: rosbag 
## ver lo de subscriber
##cada vez que llegue una imagen ejecutas callback
        self.sub = rospy.Subscriber('kitti/camera_color_left/image_raw', Image, self.callback)
        #self.sub = rospy.Subscriber('/stereo_camera/left/image_rect_color', Image, self.callback)
	

#        self.sub=rospy.Subscriber('/home/franz/Descargas/project_video.mp4'.Image,self.callback)


    def callback(self, message):
        global pub,flag,color_transform,contador,tiempo
        msg=message #cualquier cosa, publico la imagen original

        pub_orig.publish(msg)
        if not flag:
            flag=True
            # #Lectura de imagen
            start=time.time()


            time2 = msg.header.stamp
            #image = self.bridge.imgmsg_to_cv2(msg,"rgb8")
            #image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            image = self.bridge.imgmsg_to_cv2(msg,"rgb8")

            cv2.imwrite('ORIGINAL.jpeg',image)
            
            #image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

            #width,height = image.size

            #height=375
            #width=480




            # num_1=480/height - 1 #Este es para la altura
            # print(num_1)
            # num_2=int(width-(width/1+num_1)) #Este es para el ancho
            # print(num_2)
            # cut = width - num_2
            # print(cut)
            #image = image[:, 250:-250]

            #cv2.imwrite('ORIGINAL_CORTADA.jpeg',image)


            image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)



            cv2.imwrite('original'+str(time2)+'.jpeg',image)
            image= torch.from_numpy(image)
            image=image.float()/255
            image=image.permute(2,0,1).unsqueeze(0)
           
            # time3 = msg.header.stamp
            # cv2.imwrite(''+str(time3)+'.jpeg', image)
            #carga a la red
            with torch.no_grad():
                Input=Variable(image).cuda()
                output=self.model(Input)
            #Convertir a color
            color=output[0].cpu().max(0)[1].data.unsqueeze(0)
            color=color.numpy()
            color=color[0,:,:]
            
            
            #start=len(color)*0.75   #prueba 1
            start_2=int(len(color)*0.75)
            start=int(len(color)*0.65)
            #start=int(start)
            canales = 0
            canal_act=0
            canal_alt=0
            conteo_act = 0
            conteo_alt = 0
            #limit=100 Test.bag
            #limit_act=100

            limit=60
            limit_act=60 
            ident_1 = False
            ident_2 = False
            ident_3 = False
            ident_4 = False
            ident_5 = False
            prom_alt=0
            prom_act=0
            prom_canales=0

            #frames=15

            
            i=0
            
            for i in range (len(color[0])-1):
                if color[start,i]==13 and color[start,i+1]==13:
                    conteo_act +=1
                if conteo_act > limit_act:
                    conteo_act=0
                    if canal_act !=0:
                        canal_act=canal_act
                    else:
                        if ident_1==False:
                            canal_act=1
                            ident_1=True
                        elif ident_2==False:
                            canal_act=2
                            ident_2 = True
                        elif ident_3==False:
                            canal_act=3
                            ident_3=True
                        elif ident_4==False:
                            canal_act=4
                            ident_4=True
                        elif ident_5==False:
                            canal_act=5
                            ident_5=True
                elif color[start,i]==13 and color[start,i+1] !=13:
                    conteo_act = 0         

                if color[start,i]==14 and color[start,i+1]==14:
                    conteo_alt +=1
                if conteo_alt > limit:
                    conteo_alt=0
                    if ident_1==False:
                        canal_alt=1
                        ident_1=True
                    elif ident_2==False:
                        canal_alt=2
                        ident_2 = True
                    elif ident_3==False:
                        canal_alt=3
                        ident_3=True
                    elif ident_4==False:
                        canal_alt=4
                        ident_4=True
                    elif ident_5==False:
                        canal_alt=5
                        ident_5=True
                elif color[start,i]==14 and color[start,i+1] !=14:
                    conteo_alt = 0   
            if canal_act !=0:
                canales = 1+canal_alt    
            else:
                canales = canal_alt

            prom_act=canal_act
            prom_alt=canal_alt
            prom_canales=canales



            i=0
            conteo_act = 0
            conteo_alt = 0
            canales = 0
            canal_act=0
            canal_alt=0
            ident_1 = False
            ident_2 = False
            ident_3 = False
            ident_4 = False
            ident_5 = False


            for i in range (len(color[0])-1):
                if color[start_2,i]==13 and color[start_2,i+1]==13:
                    conteo_act +=1
                if conteo_act > limit_act:
                    conteo_act=0
                    if canal_act !=0:
                        canal_act=canal_act
                    else:
                        if ident_1==False:
                            canal_act=1
                            ident_1=True
                        elif ident_2==False:
                            canal_act=2
                            ident_2 = True
                        elif ident_3==False:
                            canal_act=3
                            ident_3=True
                        elif ident_4==False:
                            canal_act=4
                            ident_4=True
                        elif ident_5==False:
                            canal_act=5
                            ident_5=True
                elif color[start_2,i]==13 and color[start_2,i+1] !=13:
                    conteo_act = 0         

                if color[start_2,i]==14 and color[start_2,i+1]==14:
                    conteo_alt +=1
                if conteo_alt > limit:
                    conteo_alt=0
                    if ident_1==False:
                        canal_alt=1
                        ident_1=True
                    elif ident_2==False:
                        canal_alt=2
                        ident_2 = True
                    elif ident_3==False:
                        canal_alt=3
                        ident_3=True
                    elif ident_4==False:
                        canal_alt=4
                        ident_4=True
                    elif ident_5==False:
                        canal_alt=5
                        ident_5=True
                elif color[start_2,i]==14 and color[start_2,i+1] !=14:
                    conteo_alt = 0   
            if canal_act !=0:
                canales = 1+canal_alt    
            else:
                canales = canal_alt

            prom_act+=canal_act
            prom_alt+=canal_alt
            prom_canales+=canales

            
            canal_act=int(prom_act/2)
            canal_alt=int(prom_alt/2)
            canales=int(prom_canales/2)
            print("Canal_Alt")    
            print(canal_alt)
            print("Canal_Act")    
            print(canal_act)
            print("Canales")    
            print(canales)


            # print("Canal_Alt")    
            # print(canal_alt)
            # print("Canal_Act")    
            # print(canal_act)
            # print("Canales")    
            # print(canales)
            #print(start)




            #pub.publish(canal_alt)
            #pub.publish(canal_act)
            #pub.publish(canales)

            color=color_transform(color)

            #cv2.imwrite('image.jpeg', color)
            #img_array.append(color)

            #out = cv2.VideoWriter('video_project.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
            time2 = msg.header.stamp

            #color = cv2.resize(color, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(''+str(time2)+'.jpeg',color)
            

            color=self.bridge.cv2_to_imgmsg(color,"rgb8")

            

            pub.publish(color)
            tiempo += time.time()-start
            contador +=1
            if contador==200:
                print tiempo/contador
            flag=False

    #for i in range(len(img_array[i])):
    #    out.write()


node=resend()

while not rospy.is_shutdown():
    continue


