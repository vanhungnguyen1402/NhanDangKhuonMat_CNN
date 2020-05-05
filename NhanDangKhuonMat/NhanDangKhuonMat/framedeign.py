import cv2
import matplotlib.pyplot as plt 
from tkinter import filedialog
from tkinter import *
import numpy as np 
# PIL là thư viện hình ảnh của python
from PIL import Image
from PIL import ImageTk
import tensorflow as tf 
from sklearn.model_selection import train_test_split

IMG_SIZE =32
testing_data =np.load('testing_data.npy')
training_data =np.load('training_data.npy')

new_model =tf.keras.models.load_model('model')
new_model.summary()


train,test = train_test_split(training_data,test_size = 0.2)


X_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1) #train co 2  phan x[0] la anh 
# x[1] la phan loai phai anh face khong array([1,0]) 
y_train = [i[1] for i in train] #

X_test = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1) #
y_test = [i[1] for i in test] #


loss, acc = new_model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

class_name =[' Nguyen Van Hung','Nguyen Duc Huynh','dog','cat','unkown']

fig = plt.figure(figsize=(16,12))
for num,data in enumerate(training_data[:16]):
    im_num = data[1]
    img_data = data[0]
    y= fig.add_subplot(4,4,num+1)
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    model_out = new_model.predict([data])[0]
    if np.argmax(model_out) ==0:
        #str_label ='noface'
        #plt.title(" face {:.2f}  noface {:.2f}".format(float(model_out[0]),float(model_out[1]))) 
        plt.title("Hung Den {:.2f}".format(float(model_out[0])))
    elif np.argmax(model_out)==1:
        #str_label = 'face'
        #plt.title(" face {:.2f}  noface {:.2f}".format(float(model_out[0]),float(model_out[1])))
        plt.title("Huynh {:.2f}".format(float(model_out[1])))
    elif np.argmax(model_out)==2:
        plt.title("dogg {:.2f}".format(float(model_out[2])))
    elif np.argmax(model_out) ==3:
        plt.title("catt {:.2f}".format(float(model_out[3])))
    else:
        plt.title("Unkown {:.2f}".format(float(model_out[4])))
    y.imshow(orig,cmap = 'gray')
    #plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

test_precent =0
test_name =''
a1= ''
def open_foder():
    test_name =''
    img = Button(face,width=547, height=380, border=0, background='#DEEBF7', activebackground='#DEEBF7')
    #logo =camera()
    #path =filedialog.askopenfilename(initialdir="/",title ="Select file",filetypes =(("jpeg files","*.jpg"),("all files ","*.*")))
    path = filedialog.askopenfilename()
    print(len(path))
    #if len(path)>0:

    a1 =plt.imread(path)
    
    img2 =Image.fromarray(a1)

    img1 = ImageTk.PhotoImage(img2)

    a2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    a2 = cv2.resize(a2,(IMG_SIZE,IMG_SIZE))
    data = a2.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    
    prediction =new_model.predict([data])[0]
    print(np.argmax(prediction))
    test_name =str(class_name[np.argmax(prediction)])
    print(test_name)
    check_title = Button(face, text=test_name, background='#DEEBF7', activebackground='#DEEBF7', width=15, border=0, font = ("Helvetica 15 italic"))
    check_title.place(x=423, y=81)

    img.image = img1
    img.configure(image =img1)
    img.place(x=252, y=137)
    frame1.destroy()
    frame2.destroy()


def check_button():
    test_name =''
    img = Button(face,width=547, height=380, border=0, background='#DEEBF7', activebackground='#DEEBF7')


    if (len(a1)>0):

        a2  = plt.imread(a1)
        plt.imshow(a2)
        plt.show(a2)
        a2 = Image.fromarray(a2)


        
        

    

    
    



#	Tệp tin XML  Haarcade dùng để nhận diện khuôn mặt trong một bức ảnh, là dữ liệu được sử dụng phôt biến nhất
def camera():

    faceCascade = cv2.CascadeClassifier('F:/NhanDangKhuonMat/NhanDangKhuonMat/haarcascade/haarcascade_frontalface_default.xml')
    #faceCascade = cv2.CascadeClassifier('Haar Cascade/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    id =0 
    
    font =cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cap.read()
        
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20),
            flags =cv2.CASCADE_SCALE_IMAGE
        )
        #print("face {:.2f} noface {:.2f}".format(float(model_out[0]),float(model_out[1])))
        #print("Found {0} face ".format(len(faces)))
        roi_color = img
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            


            roi_color = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
            roi_color = cv2.resize(roi_color,(IMG_SIZE,IMG_SIZE))
            data = roi_color.reshape(-1,IMG_SIZE,IMG_SIZE,1)
            model_out = new_model.predict([data])[0]
            
            a =new_model.predict([data])
            print(a)

            if np.argmax(model_out)==0:
            #confidence =" {0}%".format(round(100-confidence))
                id ='Hung Den'
                print('Hung Den')
                confidence = "{:.2f}%".format(model_out[0])

            elif np.argmax(model_out)==1:
            #confidence =" {0}%".format(round(100-confidence))
                id = 'Huynh'
                print('Huynh')
                confidence = "{:.2f}%".format(model_out[1])
            elif np.argmax(model_out) ==2:
                id ='dog'
                print('dog')
                confidence = "{:.2f}%".format(model_out[2])
            elif np.argmax(model_out) ==3:
                id ='cat'
                print('catt')
                confidence = "{:.2f}%".format(model_out[3])
            else:
                id ='Unkown'
                confidence ="{:.2f}%".format(model_out[4])
            print("Hung Den {:.2f} Huynh {:.2f} dogg {:.2f} catt {:.2f} Unkown {:.2f}".format(
                float(model_out[0]),float(model_out[1]),float(model_out[2]),float(model_out[3]),float(model_out[4])))
            
            
            cv2.putText(
                img,
                str(class_name[np.argmax(model_out)]),
                (x+5,y-5),
                font,
                1,
                (255,255,255),
                2
            )
          
        cv2.imshow('video',img)
        k = cv2.waitKey(30) & 0xff
        if k == 13: # press 'Enter' to quit
            break
    cap.release()

    cv2.destroyAllWindows()
    plt.show()
    cv2.waitKey(0)

    

##       THIẾT KẾ GUI        ####



root = Tk()
#create the root window
face = Frame(root)
face.grid()
 

root.title("face recognition ")
root.geometry("850x550")
root.configure(background='#DEEBF7')
backg = Button(face, height=850, width=550, background='#DEEBF7',
    activebackground='#DEEBF7', border=0)
backg.pack()
# Trong Tkinter có hai loại widget, loại thứ nhất là các widget thường như nút bấm, textbox… 
# loại thứ hai là containter, đây là những widget chứa các widget khác, chúng còn có một cái tên nữa là layout. 
# Trong Tkinter có 3 loại layout là pack, grid và place.
# place là kiểu layout tự do, thuật ngữ gọi là Absolute Positioning, 
# tức là bạn sẽ phải tự quy định vị trí cũng như kích thước của các widget.
# Layout pack sắp xếp các widget của bạn theo chiều ngang và chiều dọc.
# layout grid sắp xếp widget theo dạng bảng…

title = Label(face, text = "FACE RECOGNITION", background = '#DEEBF7', font = ("Helvetica 24 bold"))
title.place(x=250, y=40)


frame1=Frame(root, width=551, height=384, background='#000000')
frame1.configure()
frame1.place(x=250, y=135)
frame2=Frame(root, width=547, height=380, background='#DEEBF7', borderwidth=5)
frame2.place(x=252, y=137)

operfoder = Button(face, text = "Forder", height=2, width=20, background='#5F85CD',
    borderwidth=5, activebackground='#A2C7E8', relief='raised', command=open_foder)
operfoder.place(x=60, y=150)

check = Button(face, text = "CHECK", height=2, width=20, background='#5F85CD',
    borderwidth=5, activebackground='#A2C7E8', relief='raised', command=check_button)
check.place(x=60, y=250)
check.destroy()

cameras = Button(face,text = "CAMERA",height =2,width =20,background ='#5F85CD',
    borderwidth =5,activebackground ='#A2C7E8',relief ='raised',command = camera)
cameras.place(x =60,y =350)






back = Button(face, text="CANCEL", height=2, width=20, background='#FF5559',
    borderwidth=5, activebackground='#BF230A', relief='raised', command = root.destroy)
back.place(x=60, y=450)


root.mainloop()
#kick off the window's event-loop
# rootlà đối tượng có mainloop phương thức Nói một cách đơn giản hơn, đây chỉ là cách tkinter hoạt động - 
# bạn luôn kết thúc tập lệnh của mình bằng cách gọi mainloopphương thức của cửa sổ gốc