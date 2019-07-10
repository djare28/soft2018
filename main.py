import cv2
import math
from skimage import color
from skimage.measure import label
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
import glob, os
from sklearn.datasets import fetch_mldata





def dot(v,w): 
    X,Y=w
    x,y=v
    r = x*X+y*Y
    return r

def vector(b,e):
    X,Y=e
    x,y=b
    r = (X-x,Y-y)
    return r

def length(v):
    x,y=v
    r = math.sqrt(x*x+y*y)
    return r

def add(v,w):
    X,Y=w
    x,y=v
    r = (x+X,y+Y)
    return r

def dodajBroj(br):
    global suma
    suma+=br

def distance(p0,p1):
    r = length(vector(p0,p1))
    return r

def scale(v,sc):
    x,y=v
    r = (x*sc,y*sc)
    return r

def unit(v):
    mag=length(v)
    x,y=v
    r = (x/mag,y/mag)
    return r


"""http://www.fundza.com/vectors/point2line/index.html"""
def pnt2line2(pnt, start, end):
    pnt_vec = vector(start, pnt)
    line_vec = vector(start, end)
    line_len = length(line_vec)

    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    line_unitvec = unit(line_vec)
    t = dot(line_unitvec, pnt_vec_scaled)
    e = 1
    if t > 1.0:
        t = 1.0
        e = -1
    elif t < 0.0:
        t = 0.0
        e = -1

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    r = (dist, (int(nearest[0]), int(nearest[1])), e)
    return r
    







def houghTransformation(frejm,rgb2gray):
    minimum=50
    maksimum=150
    x0=1000
    y0=1000
    y1=-1000
    x1=-1000
    
    retIvica=cv2.Canny(rgb2gray,minimum,maksimum,3)
    lines=cv2.HoughLinesP(retIvica,1,np.pi/180,40,100,8) 

    i = 0
    while (i < len(lines)): 
        x01 = lines[i][0][0]
        y01 = lines[i][0][1]
        y02 = lines[i][0][3]
        x02 = lines[i][0][2]
        i = i + 1
        
        if  x02 > x1:
            x1 = x02
            y1 = y02
        if  x01 < x0:
            y0 = y01
            x0 = x01

    print("Koordinate linije: (",x0,",",y0,"), (",x1,",",y1,").")
    return x0,y0,x1,y1

def hough(video):
    
    
    capture = cv2.VideoCapture(video)
    kernel = np.ones((2,2),np.uint8)
    
    while(capture.isOpened()):
        ret, frejm = capture.read()
        
        rgb = cv2.cvtColor(frejm, cv2.COLOR_BGR2RGB)
        rgb2gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb2gray = cv2.dilate(rgb2gray,kernel)
        
        capture.release()
        frejmTemp=frejm
        cv2.destroyAllWindows()
        r = houghTransformation(frejmTemp,rgb2gray)
        return r


unetVideo=input("Uneti broj video snimka: video-")
suma=0
putanja="C:\\Users\\darko\\Desktop\\Untitled Folder\\videos\\video-"
video=putanja+unetVideo+".avi"
video0=cv2.VideoCapture(video)
     
x1,y1,x2,y2=hough(video)

retIvica = [(x1, y1), (x2, y2)]

data_home_location = 'C:\\Users\\darko\\Desktop\\Untitled Folder\\scikit_learn_data\\'
mnist=fetch_mldata('MNIST original',data_home=data_home_location) 
mnist_brojevi=[]


id = -1    
def IdIncrement():
    global id
    id = id + 1
    print('ID: ', id)
    return id

maksimalnoRastojanje=15    
def maksimalnaDistanca(broj,brojevi) :
    rez = []
   
    for br in brojevi:
         
          if (maksimalnoRastojanje > distance(broj['sredina'],br['sredina'])) :

            rez.append(br)

    return rez
        

def preparePicture(slikaCB) :
    
    slika = np.zeros((28,28),np.uint8)
    xM  = 1000
    xV  = -10
    yM  = 1000
    yV  = -10
    
    z = 0
    sirina = 0
    visina = 0

    try :
        labelaSlika  = label(slikaCB)
        regioni = regionprops(labelaSlika)    
        while (len(regioni) > z):
            promenljiva = regioni[z].bbox
            if xM > promenljiva[0]:
                xM = promenljiva[0]
            if yM > promenljiva[1]:
                yM = promenljiva[1]
            if xV > promenljiva[2]:
                xV = promenljiva[2]
            if yV > promenljiva[3]:
                yV = promenljiva[3]
            z = z + 1

        sirina = xV - xM    
        visina = yV - yM    
        slika [0 : sirina, 0 : visina] += slikaCB[ xM : xV, yM : yV]
        return  slika
    except  ValueError: 
        print ("catch")    
        pass

        
def nearest (lista,elem):
    vr = lista[0]
    
    l0 = distance(elem['sredina'],vr['sredina'])
    for el in lista:
        l1 = distance(elem['sredina'],el['sredina'])
        if l0 > l1:
            vr = el
    return vr




def findNumber(slika) :
    
    faktor = 0.80
    
    slikaCB=((color.rgb2gray(slika)/255.0)>faktor).astype('uint8')
    slika  =  preparePicture(slikaCB)

    rez  =  -1
    minSuma  =  1000
    l = len(mnist_brojevi)
    
    for i in range(l) : 
        suma  =  0
        mnist_slika  =  mnist_brojevi[i]
        suma  =  np.sum(mnist_slika != slika)
       
        if minSuma > suma :
            minSuma  =  suma
            #print('RAZLIKE: ',suma)
            rez  =  mnist.target[i]
        i = i + 1
    return  rez
 
    

    
def main():
    
    trezultati = [74, 84, 78, 102, 121, 83, 96, 75, 126, 100]

    brojevi = []
    frejm = 0
    
    dgr = 160
    ggr = 255
    
    reshape_faktor = 28
    i = 0
    while (i < 70000):
        slika  =  mnist.data[i].reshape(reshape_faktor,reshape_faktor)
        faktor = 0.80
        picture  =  ((color.rgb2gray(slika)/255.0)>faktor).astype('uint8')
        picture  =  preparePicture(picture)
        
        mnist_brojevi.append(picture)
        i = i + 1
    
    kernel = np.ones((2,2),np.uint8)
    
    dgr1 = np.array([dgr , dgr , dgr],dtype = "uint8")
    ggr1 = np.array([ggr , ggr , ggr],dtype = "uint8")
    
    while(1): 
        ret, slika = video0.read()
        if not ret: break
        
        maska = cv2.inRange(slika, dgr1, ggr1) 
        
        slikaCB = 1.0 * maska
        slikaCB2 = 1.0 * maska
        
        slikaCB = cv2.dilate(slikaCB,kernel)
        
        
        slikaCBLabel,niz = ndimage.label(slikaCB)
        objekti = ndimage.find_objects(slikaCBLabel)
        j = range(niz)
        for i in j: 
            pozicija = objekti[i]
            duzina = []
            sredina = []
           
            duzina.append(pozicija[1].stop - pozicija[1].start)           
            duzina.append(pozicija[0].stop - pozicija[0].start)

            sredina.append(pozicija[1].start/2 + pozicija[1].stop/2)
            sredina.append(pozicija[0].start/2 + pozicija[0].stop/2)

            
            if duzina[0] >= 9 or duzina[1] >= 9 : 
                broj = {'frejm' : frejm, 'sredina' : sredina, 'duzina' : duzina}
               
                rezultat = maksimalnaDistanca(broj,brojevi)
                
                lnrez=len(rezultat)
                half_size = 14
                if lnrez == 0 :
                   
                    x11 = sredina[0] - half_size
                    y11 = sredina[1] - half_size
                    y22 = sredina[1] + half_size
                    x22 = sredina[0] + half_size
                    broj['prosao'] = False                      
                    broj['id'] = IdIncrement()
                    broj['vrednost'] = findNumber(slikaCB2[int(y11):int(y22),int(x11):int(x22)])
                    brojevi.append(broj)
                    print ("Naisao je broj: " + format(int(broj['vrednost'])))
                else:
                    
                    br = nearest(rezultat,broj)
                    br['frejm'] = broj['frejm']
                    br['sredina'] = broj['sredina']
                    
        for br in brojevi :
            frameDiff  =  frejm - br['frejm']
            if ( frameDiff < 3 ): 
                dist, pnt,r  =  pnt2line2(br['sredina'],retIvica[0],retIvica[1])
                if r  >  0 :
                    if dist <= 9 :
                        if br['prosao'] == False:
                            (x,y) = br['sredina']
                            br['prosao'] = True
                            print ("+++Prosao je broj: " + format(int(br['vrednost'])))
   
                            dodajBroj(br['vrednost'])
      
        frejm = frejm + 1
      
   
    print ("Suma brojeva: " + format(int(suma)))
    print ("Tacan rezultat: " + format(trezultati[int(unetVideo)]))
    if trezultati[int(unetVideo)] < int(suma) :
        print ("Procenat uspesnosti: " + format(trezultati[int(unetVideo)]*100/int(suma)))
    else :
        print ("Procenat uspesnosti: " + format(int(suma)*100/trezultati[int(unetVideo)]))
    video0.release()
main()
