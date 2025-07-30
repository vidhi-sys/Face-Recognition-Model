import numpy as np 
import sklearn
import pickle
import cv2

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('model_svm.pickle', 'rb'))
pca_models = pickle.load(open('pca_dict_pickle', 'rb'))
model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']

def face_recognition(file_name):
    img = cv2.imread(file_name)  # RGB
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    predictions = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_img = gray_img[y:y+h, x:x+w]
        
        # Normalization bringing into 0 to 1 scale
        norm_img = crop_img / 255.0
        # Resize
        resized_img = cv2.resize(norm_img, (100, 100))
        
        # Flattening
        flat_img = resized_img.flatten()
    
        # get eigen image
        eigen_img = model_pca.transform(flat_img.reshape(1, -1))
        inv_img = model_pca.inverse_transform(eigen_img)
        
        prob_score = model_svm.predict_proba(eigen_img)
        gender = 'female' if prob_score[0][0] > 0.5 else 'male'
        
        # Define colors for male (blue) and female (pink) in BGR format
        male_color = (0, 0, 255)
        female_color = (255, 105, 180)
    
        # Determine color and text based on predicted gender
        color = male_color if gender == 'male' else female_color
        text = 'Male' if gender == 'male' else 'Female'
    
        # Draw the rectangle and text on the image
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 8)
        cv2.putText(img_copy, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)
    
        # Append a dictionary for each face
        predictions.append({
            'gender': gender,
            'prob_score': prob_score.max(),
            'eigen_img': eigen_img,
            'crop_img': crop_img
        })
    
    return img_copy, predictions