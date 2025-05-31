# Surface-Defect-Detection-in-Steel-Sheets

Steel surface defect detection is crucial for maintaining product quality and ensuring efficient manufacturing processes. Conventional inspection methods are laborious, prone to errors, and not scalable since they rely on manual observation or traditional computer vision algorithms. This project proposes a ResNet-18-based automated deep learning method for precise steel surface fault identification and categorization. Models are trained and evaluated using the NEU Surface Defects Database, which includes six defect types: scratches, pitted surface, rolled-in scale, patches, inclusions, and crazing. 

Data augmentation methods, including rotation, flipping, and contrast adjustment, are used to improve model generalization. Grad-CAM was utilized to enhance interpretability by visualizing the location of defects. The suggested ResNet-18 model performs better than traditional CNN architectures, locating defect more accurately and effectively. ResNet-18 is a reliable and scalable steel quality control solution that supports intelligent manufacturing and avoids defects when compared to traditional methods.

Based on the analysis of images of steel surface, the model will label them as either defect or non defect. Users can upload pictures for real-time diagnosis through the system, which is designed as a web application using Flask. This method facilitates precision steel manufacturing, promotes sustainable manufacturing, and enhances early defect detection.

# Proposed Methodology

**Deep Learning Model**:Utilized ResNet-18, a convolutional neural network (CNN), for accurate classification of steel surface defects.

**Dataset**:Used the NEU Surface Defect Database, which contains high-resolution images of size 200x200 pixels across six defect types: scratches, rolled-in scale, pitted surface, patches, inclusions, and crazing.
![Dataset](https://github.com/user-attachments/assets/721559e0-acc3-41cd-8b9c-f5e9e85fe22f)

**Preprocessing Techniques**:Image normalization, resizing, noise reduction, and data augmentation
*Resizing* all images to fit model input requirements.

*Image Normalization* to scale pixel values between 0 and 1.

*Noise Reduction* to remove unwanted pixel artifacts.

*Data Augmentation* to increase dataset diversity:
Rotation

Horizontal/Vertical Flipping

Brightness Adjustment

Contrast Enhancement

Ensures model generalization and robustness under real-world conditions

**Model Training**:Applied transfer learning with hyperparameter tuning for optimal performance. Compared with VGG16 and EfficientNet for accuracy and efficiency.

**Deployment**:Integrated the trained model into a Flask-based web application for real-time prediction and user interaction.

# System Architecture
![image](https://github.com/user-attachments/assets/bad8621c-61fa-4437-8738-606883e6fb25)

