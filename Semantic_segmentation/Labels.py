from PIL import Image
import torch
import numpy as np
import glob
from argparse import ArgumentParser
from utils.transform import Colorize
import cv2
import sys
import shutil
from torchvision.transforms import ToPILImage


#copyfile(src, dst)

def labels_change(in_img,in_img2,in_img3):

    #Lo primero que se hace es transformar las imagenes en una matriz
    array=np.array(in_img) #Matriz 1 
    array_2=np.array(in_img2) #Matriz 2 
    array_3=np.array(in_img3) #Matriz 3
    
    no_label = 0 #Esto es un identificador de cuando se ve algo que no se puede identificar con el programa, es decir
    #no lineas, no carros no peatones

    # for i in range(len(array)):
    #     a=False
    #     b=False
    #     inicio=0
    #     fin=0
    #     for j in range(1279):
    #         if array[i,j]==3 and array[i,j+1]==0:
    #             inicio=j+1
    #             a=True
    #         if array[i,j]==0 and array[i,j+1]==3:
    #            fin = j+1
    #            b=True
    #         if a and b and (fin-inicio)<50:
    #             array[i,inicio:fin]=3    


    np.place(array_2,array_2==1, 20) #Aca se cambia el identificador de la imagen de la matriz 2
    np.place(array_2,array_2==2, 40) #En este se identifica el 1 como lineas continuas y el 2 como lineas segmentadas

    #Entonces se cambia, lineas continuas ahora es 20 y lineas segmentadas es 40
    #La idea es sumar las 2 imagenes
    #como en la imagen 1 se usaban los siguientes identificadores, puedes sumar las imagenes punto a punto

    #negro 0
    #acera 1 
    #doble blanco 2 
    #blanco 3 
    #doble amarillo 4
    #amarillo 5
    #doble otro 6
    #otro 7
    #paso peatonal 8


    array=array+array_2 #Se suman las imagenes 
    #Generamos nuevos identificadores
    np.place(array,array==21,1) #1 Acera
    np.place(array,array==22,2) #2 Lineas doble blancas continuas 
    np.place(array,array==23,3) #3 Lineas blancas continuas
    np.place(array,array==24,4) #4 Lineas doble amarillas continuas
    np.place(array,array==25,5) #5 Lineas amarillas continuas
    np.place(array,array==26,no_label) #6 Estas no las quiero porque son las de otro tipo de lineas y en verdad me van a 
    np.place(array,array==27,no_label) #7 generar ruido
    np.place(array,array==42,6) #8 Lo mismo de arriba pero segmentado
    np.place(array,array==43,7) #9
    np.place(array,array==44,8)#10
    np.place(array,array==45,9)#11
    np.place(array,array==46,no_label) #12 Lo mismo de las otras lineas
    np.place(array,array==47,no_label) #13
    np.place(array,array==48,no_label) #14

#Modificar 20,28,40,41

    np.place(array,array==20,no_label) #13
    np.place(array,array==28,no_label) #14
    np.place(array,array==40,no_label) #13
    np.place(array,array==41,1) #14
            
    for i in range(len(array)):
        a=False
        b=False
        c=False
        d=False 
        e=False
        f=False               
        g=False
        h=False
        inicio=0
        fin=0
        inicio_2=0
        fin_2=0
        inicio_3=0
        fin_3=0
        inicio_4=0
        fin_4=0

        for j in range(1279):
            if array[i,j]==3 and array[i,j+1]==0:
                inicio=j+1
                a=True
            if array[i,j]==0 and array[i,j+1]==3:
               fin = j+1
               b=True
            if a and b and (fin-inicio)<50:
                array[i,inicio:fin]=3

            if array[i,j]==7 and array[i,j+1]==0:
                inicio_2=j+1
                c=True
            if array[i,j]==0 and array[i,j+1]==7:
               fin_2 = j+1
               d=True
            if c and d and (fin_2-inicio_2)<50:
                array[i,inicio_2:fin_2]=7

            if array[i,j]==5 and array[i,j+1]==0:
                inicio_3=j+1
                e=True
            if array[i,j]==0 and array[i,j+1]==5:
               fin_3 = j+1
               f=True
            if e and f and (fin_3-inicio_3)<50:
                array[i,inicio_3:fin_3]=5

            if array[i,j]==9 and array[i,j+1]==0:
                inicio_4=j+1
                g=True
            if array[i,j]==0 and array[i,j+1]==9:
               fin_4 = j+1
               h=True
            if g and h and (fin_4-inicio_4)<50:
                array[i,inicio_4:fin_4]=9                

    #Tengo una nueva imagen con esas nuevas etiquetas

    #Estas son las etiquetas de leo que las cambio de nuevo para sumarlas con lo de el
    #Tengo un problema con los peatones por eso lo tengo comentado
    #Cambiamos los numeros de las etiquetas 

    np.place(array_3,array_3==0,10)    
    np.place(array_3,array_3==1,20) #vehiculos S 
    np.place(array_3,array_3==2,30) #vehiculos M 
    np.place(array_3,array_3==3,40) #vehiculos L 
    np.place(array_3,array_3==4,50) #Carril actual 
    np.place(array_3,array_3==5,70) #Carril alternativo
    np.place(array_3,array_3==6,0)


    #np.place(array_3,array_3==0,80) #15    Esto era lo comentado
    
    array=array+array_3 #Sumamos las imagenes de nuevo


    # np.place(array,array>20 && array<30,20)
    # np.place(array,array>30 && array<40,30)
    # np.place(array,array>40 && array<50,40)

    #Aca la idea es que como yo prefiero que muestre los carros que las lineas, cuando se suman lineas con carros da carros
    #por eso 20 (carro pequeno) + 3 (linea blanca continua) es reemplazada por el carro = 20

    np.place(array,array==11,10)
    np.place(array,array==12,10)
    np.place(array,array==13,10)
    np.place(array,array==14,10)
    np.place(array,array==15,10)
    np.place(array,array==16,10)
    np.place(array,array==17,10)
    np.place(array,array==18,10)
    np.place(array,array==19,10)

    np.place(array,array==21,20) 
    np.place(array,array==22,20)
    np.place(array,array==23,20)
    np.place(array,array==24,20)
    np.place(array,array==25,20)
    np.place(array,array==26,20)
    np.place(array,array==27,20)
    np.place(array,array==28,20)
    np.place(array,array==29,20)

    #Asi con todos los carros, en este caso medianos

    np.place(array,array==31,30)
    np.place(array,array==32,30)
    np.place(array,array==33,30)
    np.place(array,array==34,30)
    np.place(array,array==35,30)
    np.place(array,array==36,30)
    np.place(array,array==37,30)
    np.place(array,array==38,30)
    np.place(array,array==39,30)

    #En este caso grandes

    np.place(array,array==41,40)
    np.place(array,array==42,40)
    np.place(array,array==43,40)
    np.place(array,array==44,40)
    np.place(array,array==45,40)
    np.place(array,array==46,40)
    np.place(array,array==47,40)
    np.place(array,array==48,40)
    np.place(array,array==49,40)


    #El 50 es el carril actual en el que va el carro, por eso es importante que se vean las lineas
    #Por eso 50(carril actual) + 1(acera) = 1, porque quiero que se vea la acera 

    np.place(array,array==51,1)
    np.place(array,array==52,2)
    np.place(array,array==53,3)
    np.place(array,array==54,4)
    np.place(array,array==55,5)
    np.place(array,array==56,6)
    np.place(array,array==57,7)
    np.place(array,array==58,8)
    np.place(array,array==59,9)

    #igual para carriles alternativos

    np.place(array,array==71,1)
    np.place(array,array==72,2)
    np.place(array,array==73,3)
    np.place(array,array==74,4)
    np.place(array,array==75,5)
    np.place(array,array==76,6)
    np.place(array,array==77,7)
    np.place(array,array==78,8)
    np.place(array,array==79,9)

    #Al final organizo esto y reemplazo 20, por 10, 30 por 11 y asi
    np.place(array,array==10,15)
    np.place(array,array==20,10)
    np.place(array,array==30,11)
    np.place(array,array==40,12)
    np.place(array,array==50,13)
    np.place(array,array==70,14)

    np.place(array,array==0,30)
    np.place(array,array==15,0)
    np.place(array,array==30,15)
 
    return Image.fromarray(array) #Y regreso la imagen


def fix_label(filenameGt,filenameGt2,filenameGt3):



    #Le pasamos las 3 imagenes de cada carpeta

#    for i in range (0,len(filenameGt)): #El largo es el numero de archivos que hay en la carpeta
    for i in range (0,len(filenameGt)): #El largo es el numero de archivos que hay en la carpeta

        for j in range(0,len(filenameGt3)): #El largo es el numero de archivos que hay en la carpeta 3, no es el mismo que el
        #tamano de la carpeta de leo, porque el uso una seleccion aleatoria de las imagenes que yo use

        #en name_1 cambio toda la raiz del directorio, porque el nombre de la imagen es tipo:
        #/home/franz/Escritorio/lanes/type/val/prueba y como no estan en la misma carpeta reemplazo esa linea del directorio por una H
        #basicamente fue la solucion que se me ocurrio

            #name_1=filenameGt[i].replace("/home/franz/Escritorio/lanes/type/val/","H")
            #name_2=filenameGt3[j].replace("/home/franz/Escritorio/Segmentation_obstacles_and_driveables_zones/labels/val/","H")

            name_1=filenameGt[i].replace("/home/franz/Escritorio/lanes/type/train/","H")
            name_2=filenameGt3[j].replace("/home/franz/Escritorio/Segmentation_obstacles_and_driveables_zones/labels/train/","H")

        #Si tienen el mismo nombre, significa que estan en la carpeta de leo y entonces puedo usar esa imagen

            if name_1==name_2:

                #Se convierten las imagenes a tipo L, ahora no recuerdo que era eso pero es una vaina de la libreria numpy
                label = Image.open(filenameGt[i]).convert('L')
                label_2= Image.open(filenameGt2[i]).convert('L')
                label_3= Image.open(filenameGt3[j]).convert('L')
                label=labels_change(label,label_2,label_3) #Paso las 3 imagenes a labels_change que es la primera funcion
                #la que esta arriba, vamos pa alla
                label.save("/home/franz/Escritorio/labels/save_prueba/"+filenameGt[i].split('/')[-1])#VOLVI
                #SE guarda la imagen en una carpeta y ahora con esa carpeta puedes reconocer todas las cosas que quieras
        print(i+1) #Lo haces para cada imagen de la carpeta inicial    
    print("listo") #Terminas de hacer el procedimiento

def show_value_label(filenameGt):
    for i in range (0,len(filenameGt)):
        label = Image.open(filenameGt[i]).convert('L') #P
        label=np.array(label)
        print(np.unique(label))
    print("listo "+str(i+1))

def show_label(filenameGt,NUM_CLASSES):
    name=filenameGt.split('/')[len(filenameGt.split('/'))-1]


    label = Image.open(filenameGt).convert('L') # P  para los q no sea lineas
    color_transform=Colorize(NUM_CLASSES)
    label=torch.from_numpy(np.array(label)).long().unsqueeze(0)
    label=label[0].data
    label_color = Colorize()(label.unsqueeze(0)) 
    label_color = ToPILImage()(label_color) 
    label_color=np.asarray(label_color)
    label_color = cv2.cvtColor(label_color, cv2.COLOR_BGR2RGB)
    cv2.imshow(name,label_color)
    cv2.moveWindow(name, 300,100)
    k=cv2.waitKey(0)
    if k == 27:
        sys.exit()
    cv2.destroyAllWindows()
    cv2.waitKey(100)

def road_label_copy(dir):
    datadir=dir+"/train/*.png"
    filename=glob.glob(datadir)
    #train
    data="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/output_seg/train/label/*.png"
    filedata=glob.glob(data)
    cambiar="dataset/bdd100k/drivable_maps/labels/train"
    orig="codigos/Labels/save/output_seg/train/label"

    dst="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/road/train/"
    print("Copiando Train_labels")
    copiados=0
    for i in range (0,len(filename)):
        f_c=filename[i].replace("_drivable_id","").replace(cambiar,orig)
        if f_c in filedata:
            shutil.copyfile(filename[i], '%s/%s' % (dst, filename[i].split('/')[-1].replace("_drivable_id",""))) 
            copiados+=1
    print("Listo Train_labels  "+str(copiados))

    datadir=dir+"/val/*.png"
    filename=glob.glob(datadir)
    #val
    print("Copiando Val_labels")
    data="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/output_seg/val/label/*.png"
    filedata=glob.glob(data)
    cambiar="dataset/bdd100k/drivable_maps/labels/val"
    orig="codigos/Labels/save/output_seg/val/label"

    dst="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/road/val/"
    copiados=0
    for i in range (0,len(filename)):
        f_c=filename[i].replace("_drivable_id","").replace(cambiar,orig)
        if f_c in filedata:
            shutil.copyfile(filename[i], '%s/%s' % (dst, filename[i].split('/')[-1].replace("_drivable_id",""))) 
            copiados+=1
    print("Listo Val_labels  "+str(copiados))

def lanes_label_copy():
    datadir="/home/leonarodo/Escritorio/tesis/dataset/bdd100k/images/100k/train/*.jpg"
    filename=glob.glob(datadir)
    #train
    data="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/lanes/labels/lanes/train/*.png"
    filedata=glob.glob(data)
    cambiar="dataset/bdd100k/images/100k/"
    orig="codigos/Labels/save/lanes/labels/lanes/"

    dst="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/lanes/images/train/"
    print("Copiando Train_labels")
    copiados=0
    for i in range (0,len(filename)):
        f_c=filename[i].replace(".jpg",".png").replace(cambiar,orig)
        if f_c in filedata:
            shutil.copyfile(filename[i], '%s/%s' % (dst, filename[i].split('/')[-1])) 
            copiados+=1
    print("Listo Train_labels  "+str(copiados))

    datadir="/home/leonarodo/Escritorio/tesis/dataset/bdd100k/images/100k/val/*.jpg"
    filename=glob.glob(datadir)
    #train
    data="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/lanes/labels/lanes/val/*.png"
    filedata=glob.glob(data)
    cambiar="dataset/bdd100k/images/100k/"
    orig="codigos/Labels/save/lanes/labels/lanes/"

    dst="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/lanes/images/val/"
    print("Copiando Train_labels")
    copiados=0
    for i in range (0,len(filename)):
        f_c=filename[i].replace(".jpg",".png").replace(cambiar,orig)
        if f_c in filedata:
            shutil.copyfile(filename[i], '%s/%s' % (dst, filename[i].split('/')[-1])) 
            copiados+=1
    print("Listo Train_labels  "+str(copiados))

#Aca se empieza :)
def main(agrs):

    NUM_CLASSES=16 #Esta es la cantidad de tipos de lineas, carros y peatones 
    datadir=agrs.dirlabel+"/*.png" #Aca se selecciona de una carpeta las imagenes (cuando corres el archivo en python lo mandas a una carpeta), si son tipo png las agrega
    filename=glob.glob(datadir) #Seleccionamos el archivo 

    #El problema es que la carpeta de imagenes que estaba pasando solo te permite reconocer estos tipos de lineas:

    #negro 0
    #acera 1 
    #doble blanco 2 
    #blanco 3 
    #doble amarillo 4
    #amarillo 5
    #doble otro 6
    #otro 7
    #paso peatonal 8

    #Por eso tuve que pasar la carpeta que esta en datadir_2 esa carpeta tiene imagenes que te permiten reconocer si las lineas 
    #son continuas o discontinuas (lo importante es que son las mismas lineas que estan en en el directorio inicial)

    #datadir_2="/home/franz/Escritorio/lanes/style/val/*.png" 
    datadir_2="/home/franz/Escritorio/lanes/style/train/*.png" 
    filename_2=glob.glob(datadir_2)

    #Finalmente todo esto hay que mezclarlo con lo que hizo leo que permite reconocer

# Class                       ID               
# Pedestrians                 0
# Small vehicles              1
# Medium vehicles             2
# Large vehicles              3
# Current lane                4
# Parallel lanes              5
# Unlabelds                   6

    #Le pasas la otra carpeta con las imagenes de leo en datadir_3, el problema es que no estan todas las imagenes que yo tengo
    #entonces hay que verificar si esa imagen esta en esa carpeta
    datadir_3="/home/franz/Escritorio/Segmentation_obstacles_and_driveables_zones/labels/train/*.png"
    filename_3=glob.glob(datadir_3)

#Nos vamos a fix_label que es la funcion mas importante porque es la que te genera las imagenes

    #fix_label(filename,filename_2,filename_3) #---> Ir a fix_label, es la segunda funcion que esta arriba
    
    #Mostrar etiquetas
    for i in range (0,len(filename)):
        show_label(filename[i],NUM_CLASSES)
    #    print i 


    #ENTER Y ESCAPE

    #Pasar label de carriles
    #road_label_copy(agrs.dirlabel)

    #Mostrar value labels
    #show_value_label(filename)

    #Unir label
    # data_road=agrs.dirlabel+'/road/'+agrs.subset+'/*.png'
    # data_seg=agrs.dirlabel+'/output_seg/'+agrs.subset+'/label/*.png'
    # file_road=glob.glob(data_road)
    # file_seg=glob.glob(data_seg)

    # contador=0
    # for i in range (0,len(file_road)):
    #     dir_out="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/new_labels/"+agrs.subset+"/"
    #     if file_road[i].split('/')[-1]==file_seg[i].split('/')[-1]:
    #         #carga de imagenes
    #         road= Image.open(file_road[i]).convert('P')
    #         road=np.array(road)
    #         seg= Image.open(file_seg[i]).convert('P')
    #         seg=np.array(seg)

    #         #union de labels
    #         indexes=(road!=6)
    #         seg[indexes]=road[indexes]

    #         #Guardado
    #         dir_out+=file_road[i].split('/')[-1]
    #         img=Image.fromarray(seg)
    #         img.save(dir_out)
    #         contador+=1
    # print(contador)
    # print("Listo")

    # dir_out="/home/leonarodo/Escritorio/tesis/codigos/Labels/save/new_labels/colors/train/"
    # print(len(filename))
    # for i in range (0,len(filename)):
    #     label = Image.open(filename[i]).convert('P')
    #     color_transform=Colorize(NUM_CLASSES)
    #     label=torch.from_numpy(np.array(label)).long().unsqueeze(0)
    #     label=label[0].data
    #     label_color = Colorize()(label.unsqueeze(0))
    #     label_color = ToPILImage()(label_color)
    #     label_color.save(dir_out+filename[i].split('/')[-1])
    #     print(i+1)

    # #Pasar label de lineas
    # lanes_label_copy()

    # label = Image.open("/home/leonarodo/Escritorio/tesis/codigos/Labels/save/lanes/v2_labels/val/b1d968b9-563405f4.png").convert('P')
    # array=np.array(label)
    # print(np.unique(array))





    
        
    

    
    
   

if __name__=='__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--dirlabel',type=str)
    parser.add_argument('--subset',type=str,default='val')
    #parser.add_argument('--n_class',type=int,required=True)

    main(parser.parse_args())
    

