<div align = "center">
  <h1>  Optimizing Mango Quality Analysis with Advanced Grading and Sorting Technology</h1>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Mango Grading is an advanced grading and sorting system designed to optimize the quality of mangoes. With cutting-edge technology and intelligent algorithms, Mango Grading can accurately grade and sort mangoes based on various quality parameters such as ripeness, color, size, texture and Thermal Image Processing. The system uses sensors and cameras to analyze each mango and assign it a grade based on predefined criteria. This ensures that only high-quality mangoes are selected for sale, while lower-quality ones are discarded or processed to delivered quickly. Mango Grading is an efficient and reliable solution for optimizing mango quality and increasing profitability for growers and distributors alike.
<br><br>
This innovative technology utilizes advanced sensors, machine learning algorithms, and computer vision to accurately sort mangoes with minimal human intervention. The result is a consistent and high-quality mango product that meets the standards of the most discerning customers. Mango Sort is a game-changer in the industry, providing a reliable solution for maximizing the value of mango crops and improving the efficiency of the production process.
<br>
### Built With
The major frameworks/libraries are used in the project.

* scikit
* Tensorflow
* Numpy, Pandas
* Matplotlib
* Pillow
* Tkinter
* PhotoBoothApp

### Prerequisites

* python
* Google Colab
* Object Detection Sensor
* OS with minimum 4GB
* Thermal Camera

### Installation

1. Install Python(https://wiki.python.org/moin/BeginnersGuide/Download)
2. Clone the repo
   ```sh
   git clone https://github.com/karthiaravinth/Optimizing-Mango-Quality
   ```
3. Install packages
   ```sh
    pip install tensorflow
    pip install numpy
    pip install pandas as pd
    pip install matplotlib
    pip install tkinter
    pip install PIL
    pip install PhotoBoothApp
    pip install scikit
   ```

<!-- ROADMAP -->
## Roadmap

- [x] Install Dependencies
- [x] Install packages
- [ ] setup normal(rgb) camera
- [ ] setup thermal camera (if you train the algorithm using thermal images)
- [ ] Image Capturing (Conveyor belt, On-field image collection and etc...)
    - [ ] Live image
    - [ ] Pre collected images
- [ ] Train the ML Model
- [ ] Then Grading the Mangoes
<!-- GETTING STARTED -->
## Getting Started
<br>
The input component consists of the thermal imaging camera that captures images of the mangoes. The camera sends the images to the image processing system for analysis. The image processing component consists of the software that processes the images captured by the camera. The software extract features such as temperature, size, and shape of the mangoes. Machine learning algorithms are then applied to classify the mangoes based on their quality and ripeness parameters.
<br><br>
The mechanical sorting component consists of the mechanical system that sorts the mangoes based on their quality and ripeness parameters determined by the image processing component. The sorting mechanism separates the mangoes into different categories based on their grades and sends them to their respective storage or packaging units. The user interface component provides a user-friendly interface for operators to monitor and control the grading and sorting process. The interface displays the results of the image processing system and the status of the sorting mechanism. Operators can make adjustment to the system parameters, such as the grading criteria and sorting mechanism settings. The database component stores and manages the data collected from the mangoes during the grading and sorting process. The database records the quality and ripeness parameters of the mangoes and their grades. The data can be used for further analysis and reporting.
<br><br>
The above components work together to create an automated system for grading and sorting mangoes based on their quality and ripeness parameters. The system is designed to be efficient, accurate, and user-friendly. The thermal imaging camera and image processing software provide accurate and detailed information about the mangoes, while the mechanical sorting mechanism ensures that the mangoes are sorted accurately and efficiently. The user interface allows operators to monitor and control the system, and the database component records and manages the data collected during the process.

<!-- USAGE EXAMPLES -->
#### Working

![image](https://github.com/karthiaravinth/Optimizing-Mango-Quality/assets/110279931/39c82f13-c227-433f-82fd-5aa288224b11)

## Modules

### 1. Data Preparation:
   - Collection of normal and thermal images from different sources such as on-field, internet, and live image capturing.
   - Capturing normal images using a standard camera on a conveyor belt from different angles and distances.
   - Capturing thermal images using a thermal imaging camera to capture temperature variations.
   - Labeling the images with corresponding grades or categories for training the machine learning model.

### 2. Data Annotation:
   - Manually labeling both thermal and normal images with corresponding grades or categories.
   - Creating a dataset of images and annotations in a machine-readable format, such as a CSV file.

### 3. Data Processing:
   - Pre-processing raw thermal images to remove noise and artifacts.
   - Color correction and resizing of normal (RGB) images for standardization.
   - Data augmentation techniques such as rotation, flipping, and cropping to increase dataset diversity.
   - Feature extraction from images using techniques like convolutional neural networks (CNNs) for normal images and temperature-based analysis for thermal images.

### 4. Detection of Pests:
   - Identification of mango nut weevils (Sternochaetus mangiferae) using thermal images and temperature variations caused by the presence of insects or pests inside the fruit.
   - Training a support vector machine (SVM) using both normal and thermal images to classify mangoes based on the presence of pests.

### 5. Model Training:
   - Using both normal and thermal images as training data.
   - Pre-processing and feature extraction from images.
   - Choosing a suitable model, such as SVM, for classification based on different grades and categories.
   - Fine-tuning the chosen model on the mango dataset using backpropagation.
   - Evaluating the model's performance using standard metrics like precision, recall, and mean average precision (mAP).

### 6. Model Deployment:
   - Optimizing the trained SVM model by adjusting parameters or using techniques like cross-validation.
   - Deploying the trained and optimized model in the grading and sorting system for classifying mangoes based on quality and ripeness parameters.

### 7. Calculation and Measurement Techniques:
   - Calculating mean RGB values of mango images for ripe, semi-ripe, and defective mangoes.
   - Measuring the number of lanes and guard lanes using relationships between their lengths and widths.

### 8. Model Evaluation and Validation:
   - Assessing the model's performance in classifying mangoes using metrics such as accuracy, precision, recall, and F1 score.
   - Validating the model's accuracy by comparing predictions with ground truth data

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to thank Mr. K. MANOJ KUMAR, M.E., Associate Professor, Head of Computer Science and Engineering department and Dr. R. MANIKANDAN, M.E, Ph.D., Associate Professor (CAS) for his invaluable guidance and encouragement in completing this project work.
<br><br>
We also like to thank all the respected faculty members and staff members of Computer Science and Engineering department and various department of our college for their direct and indirect involvement in successful completion of this project and like to express our sincere thanks and gratitude to our parents and friends for their continuous encouragement and support.


<!-- References -->
## References

1. Mango Detection using Fast R-CNN by R. Swetha and R. Sridhar
2. Quality Grading of Mangoes Using Machine Learning Techniques by Gautam. (2020)
3. Automated fruit grading using computer vision by M. Jalal
4. Image analysis techniques for automatic grading of fruit by S. M. S. Islam
5. Robotic harvesting and automated fruit grading by G. N. Tiwari.
6. Computer Vision Based Mango Grading System by Arunkumar. (2017)
7. Image Processing Techniques for Mango Fruit Detection and Recognition by Mohanty. (2016)
8. Mango Grading and Sorting by S. P. Singh and P. N. Pandey (2018)
9. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Tan and Le (2019)
10. "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau (2015)
11. Budiastra, I.W.; Punvadaria, H.K. Classification of Mango by Artificial Neural Network Based on Near Infrared Diffuse Reflectance. IFAC Proc.
12. Nandi, C.S.; Tudu, B.; Koley, C. Computer vision based mango fruit grading system. In Proceedings of the International Conference on Innovative Engineering Technologies, Bangkok, Thailand, 28–29 December 2014.
13. Wanitchang, P.; Terdwongworakul, A.; Wanitchang, J.; Nakawajana, N. Non-destructive maturity classification of mango based on physical, mechanical and optical properties. J. Food Eng. 2011.
14. Zawbaa, H.M.; Hazman, M.; Abbass, M.; Hassanien, A.E. Automatic fruit classification using random forest algorithm. In Proceedings of the 2014 14th International Conference on Hybrid Intelligent Systems, Hawally, Kuwait, 14–16 December 2014; Institute of Electrical and Electronics Engineers (IEEE): Piscataway, NJ, USA, 2015.
15. Kumar, A.; Gill, G. Computer vision based model for fruit sorting using K-nearest neighbour classifier. Int. J. Electr. Electron. Eng.
16. Nandi, C.S.; Tudu, B.; Koley, C. An automated machine vision based system for fruit sorting and grading. In Proceedings of the 2012 Sixth International Conference on Sensing Technology (ICST), Kolkata, India, 18–21 December 2012; Institute of Electrical and Electronics Engineers (IEEE): Piscataway, NJ, USA, 2013.
17. https://www.mdpi.com/2076-3417/10/17/5775
18. https://agritech.tnau.ac.in/crop_protection/mango/mango_9.html
19. Nandi, C.S.; Tudu, B.; Koley, C. A Machine Vision Technique for Grading of Harvested Mangoes Based on Maturity and Quality. IEEE Sens. J.
20. Pise, D.; Upadhye, G.D. Grading of Harvested Mangoes Quality and Maturity Based on Machine Learning Techniques. In Proceedings of the 2018 International Conference on Smart City and Emerging Technology (ICSCET), Maharashtra, India, 5 January 2018; Institute of Electrical and Electronics Engineers (IEEE): Piscataway, NJ, USA, 2018.
21. Pandey, R.; Gamit, N.; Naik, S. A novel non-destructive grading method for Mango (Mangifera Indica L.) using fuzzy expert system. In Proceedings of the 2014 International Conference on Advances in Computing, Communications and Informatics (ICACCI), Noida, India, 24–27 September 2014; Institute of Electrical and Electronics Engineers (IEEE): Piscataway, NJ, USA, 2014
22. Charoenpong, T.; Kamhom, P.; Chaninongthai, K.; Krairiksh, M.; Chamnongthai, K. Volume measurement of mango by using 2D ellipse model. In Proceedings of the 2004 IEEE International Conference on Industrial Technology, IEEE ICIT ’04, Hammamet, Tunisia, 8–10 December 2004; Institute of Electrical and Electronics Engineers (IEEE): Piscataway, NJ, USA, 2005; Volume 3.
23. "Deep Learning" by Bengio, Goodfellow, and Courville (2016)
