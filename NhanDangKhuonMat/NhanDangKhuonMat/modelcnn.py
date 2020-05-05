# Xây dựng Graph

# 1. Thêm các thư viện cần thiết
from create_data import Create_data
import numpy as np #Thư viện xử lý toán học
import cv2
import tensorflow as tf #Tensoflow Lib
import matplotlib.pyplot as plt #dùng để vẽ đồ thị
import tflearn
from tensorflow.keras import layers,models
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Activation,Flatten


from sklearn.model_selection import train_test_split
img_size =32

class modelcnn():
    def model(self):
        train_data =Create_data.create_train_data()
        test_data =Create_data.create_test_data()

        plt.imshow(train_data[5][0],cmap='gist_gray')
        plt.show()
        print(train_data[5][1])
        
        train,test = train_test_split(train_data,test_size =0.3)

        X_train  = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
        y_train =[i[1]for i in train]
        X_test = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
        y_test = [i[1]for i in test]

        X_train =X_train/255.0
        X_test = X_test/255.0
        class_names = ['hung','huynh','dog','cat','Unkown']

        fig = plt.figure(figsize=(16,12))
        for num,data in enumerate(train_data[:16]):
            im_num = data[1]
            img_data = data[0]
            y= fig.add_subplot(4,4,num+1)

            y.imshow(img_data)
            plt.title(class_names[np.argmax(im_num)])
            #plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()

        # 5. Định nghĩa model với kiểu model là Sequential( nó cho phép bạn xây dựng 1 mô hình từng lớp, mỗi lớp 
        # có trọng lương tương ứng với lớp theo sau nó.)
        # Sử dụng hàm add để thêm các layer vào vào mô hình của chúng ta
        model = models.Sequential()
        # Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3 
        # dùng hàm relu làm activation và chỉ rõ input_shape cho layer đầu tiên
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,1)))
        # Thêm Convolutional layer
        model.add(layers.Conv2D(32,(3,3),   activation = 'relu'))
        # Thêm Max pooling layer
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.MaxPooling2D((2, 2)))



        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.summary()

        # Flatten layer chuyển từ tensor sang vector
        model.add(layers.Flatten())
        # Thêm Fully Connected layer với 64 nodes và dùng hàm relu (hoặc sigmoid)
        model.add(layers.Dense(64, activation='relu'))
        # Output layer với 5 node và dùng softmax function để chuyển sang xác suất.
        model.add(layers.Dense(5, activation='softmax'))

        model.summary()

        #Note: Dence là 1 kiểu Layer, là 1 loại layer tiêu chuẩn hoạt động trong hầu hết các trường hợp
        # Trong Layer dence: tất cả các node trong lớp trước kết nối với các lớp hiện tại


        # 6. Compile model, có 2 tham số là optimizer và loss (chỉ rõ hàm loss_function nào được sử dụng, phương thức 
        # Dùng để tối ưu hàm loss function). sau đó accuracy để đánh giá độ chính xác của mạng CNN
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # 7. Thực hiện train model với data
        history = model.fit(np.array(X_train),np.array(y_train),epochs =30,
            validation_data =(np.array(X_test),np.array(y_test)))
        model.save('model')

        # 8. Vẽ đồ thị loss, accuracy của training set và validation set 
        plt.plot(history.history['acc'], label='accuracy')
        plt.plot(history.history['val_acc'], label = 'val_accuracy')
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label= 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([-2, 2])
        plt.legend(loc='lower right')

        plt.show()
        # 9. Đánh giá model với dữ liệu test set
        test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
        



               

modelcnn().model()

