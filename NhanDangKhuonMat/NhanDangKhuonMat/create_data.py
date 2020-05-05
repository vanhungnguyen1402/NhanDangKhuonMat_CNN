import numpy as np 
import os 
#Sử dụng để hiển thị thanh tiến trình với sự hỗ trợ tốt cho các vòng lặp lồng nhau bằng module tqdm
from tqdm import tqdm
import cv2
from random import shuffle
train_data ='train5' 
test_data ='test1'
img_size =32
# load tất cả các ảnh này, resize lại kích thước 32*32 rồi reshape lại thành 1 vector có 1024 phần tử hay pixel
#Kết quả thu được x_train, y_train chứa dữ liệu training/// x_text, y_text chứa dữ liệu text
class Create_data:

    

    def create_label(image_name):

        word_label = image_name[:4] # 
        if word_label =='hung':
            return np.array([1,0,0,0,0])
        elif word_label =='huyn':
            return np.array([0,1,0,0,0])
        elif word_label =='dogg':
            return np.array([0,0,1,0,0])
        elif word_label =='catt':
            return np.array([0,0,0,1,0])
        else:
            return np.array([0,0,0,0,1])
        
    @classmethod
    def create_train_data(self):
        training_data =[]
        print(len(os.listdir(train_data)))
        for img in tqdm(os.listdir(train_data)):
            path =os.path.join(train_data,img)
            img_data =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_data=cv2.resize(img_data,(img_size,img_size))
            
            # Đặt hết ảnh trong create_lable vào tập dữ liệu tranining là training_data.
            training_data.append([np.array(img_data),Create_data.create_label(img)])
        shuffle(training_data)
        np.save('training_data.npy',training_data)
        return training_data
    @classmethod
    def create_test_data(self):
        testing_data= []
        for img in tqdm(os.listdir(test_data)):
            path =os.path.join(test_data,img)
            img_number =img.split('.')[0]
            img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_data =cv2.resize(img_data,(img_size,img_size))
            testing_data.append([np.array(img_data),img_number])
            #Xáo trộn danh sách (sắp xếp lại thứ tự của các mục danh sách)
        shuffle(testing_data)
        np.save('testing_data.npy',testing_data)
        return testing_data
    

c =Create_data()
c.create_train_data()
print(c)