# Eye-movement-classification-in-video
To classify different eye movements like blink, left, right, centre, up, and down in video using convolutional neural networks.Proposed architecture employs deep neural networks that run in parallel, for both eyes separately for visual feature extractions using preprocessed dataset of different eye movement images.

DETAILED SYSTEM DESIGN

Module 1 : Face detection and eye detection in video
Face detection and eye detection is done using the dlib library and opencv. The frontal face detector function of dlib is used in order to detect the face. We also use the facial landmarks in order to detect eye from the face. The file for facial landmark detection is “shape_predictor_68_face_ landmarks.dat”. We process the video frame wise in order to detect the face and eyes. Repeat the steps till the end of the video frame wise. Convert the frame to grayscale image and use the function to detect the face. Use the facial landmark function in order to detect the eyes. The location of eyes in the facial landmark file is in points : Left eye(36,37,38,……41) and  right eye(42,43,……47).

Module 2 : Dataset
The dataset we use is eye chimera dataset which contains the 7 movements centre, left, right, up right, up left, downright, down left. For blink detection we used the open eye image dataset created by ourselves along with the open eye images  from the eye chimera dataset. For closed images we used Closed Eyes in the Wild(CEW) dataset along with some images created by ourselves. Detect the eye using the eye detection algorithm and crop the left and right eye separately.
Pre-process the eyes by resizing the images to size 50x50 and grayscaling for    faster training time. Histogram normalization is also applied in pre-processing.
Storing the pre-processed  eye movements separately  in the file for eye movement classification.

Module 3 :Blink Detection
Training of the model : The dataset for open and closed eye images contain 1654 images separately. We splitted the preprocessed image set into 80% for training and the remaining to validation and test set. The input stage consists of the two classes of eye images open and close of size 50x50. Here the CNN consists of 3 convolution stages. Each stage will be  followed by Rectified Linear Unit(ReLu). ReLu layer introduces a non-linearity to the activations. Then a max pooling layer is going to be added after the ReLu stage. After convolution, ReLu and max pooling in the third convolution layer the outputs are joined in fully connected layers. For the final layer a sigmoid activation function is used to classify to the two classes.
Testing of blink detection : Video is captured through webcam for testing the blink detection Here the video is splitted into frames and eyes are detected from the frames. The eyes are the pre-processed by resizing to 50x50 and the applying histogram normalisation and then fed into the model. The model predicts it into open or close. A memory counter is kept which counts the number of consecutive closes. When an image is predicted as open then it checks the memory counter of close and if it is greater than 0 and less than 4 then it will be considered as a blink.

Module 4:Left Right Centre detection
Training of the model : Here the input for the CNN is the 3 classes of images left ,right ,centre movements of eyes. The dataset consists of 536 preprocessed images for the 3 classes separately. This model  consists of 3 convolution stages. Each stage will be  followed by Rectified Linear Unit(ReLu). ReLu layer introduces a non-linearity to the activations. Then a max pooling layer is going to be added after the ReLu stage. Then two similar covolution stages will be also added. After convolution, ReLu and max pooling in the third convolution layer the outputs should be joined in fully connected layers. The final layer activation function is a softmax function. The class with highest value will be selected as the predicted class.
Testing left right centre : Video is captured through web cam for testing the blink detection. Here the video is splitted into frames and eyes are detected from the frames. The eyes are the preprocessed by resizing to 50x50 and the applying histogram normalisation and then fed into the model. The model will predict and give the probability value for each class. The class with highest probability value will be selected as the predicted class.

MODULE 5:Left Right centre up and down movements
In this module we add two more classes up and down movements. For up and down classes since the dataset was not so accurate we add some more images into the dataset manually. In this model we used the same architecture of the above model. The model will then be tested by using the video captured through the web camera.

RESULT

For the open and close eye classification, the CNN model was trained for 15 epochs and drop out of 0.5 was done in order to avoid over-fitting as the dataset was not
huge. For the left, right, centre eye movement classification, the CNN model was trained for 50 epochs.For the left, right, centre, top and down eye movement classification, the CNN model was trained for 200 epochs. The classification accuracy of open and close eye classification model which was used for blink detection is 98 percent. The validation accuracy for this model is 99 percent. The classification accuracy of the 3 class(left, centre, right) classification model was higher than that of the 5 class(left, centre, right,up,down) classification model. The accuracy in 3 class classification is 96 percent.The validation accuracy in this case is 98 percent.  The accuracy in 5 class classification is 90 percent. The validation accuracy in this case is 89 percent. The prediction scores from both eyes are used to obtain the final output prediction. The classification without using spectacles was more accurate than using spectacles. The histogram equalization done in the images helped to increase the accuracy as it helps to differentiate the iris and the sclera more, which plays a major role in movement classification. The classification is also depended on the lighting conditions and the quality of the camera used in the experiment.




















