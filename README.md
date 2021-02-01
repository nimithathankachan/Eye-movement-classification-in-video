# Eye-movement-classification-in-video
To classify different eye movements like blink, left, right, centre, up, and down in video using convolutional neural networks.Proposed architecture employs deep neural networks that run in parallel, for both eyes separately for visual feature extractions using preprocessed dataset of different eye movement images.


How to run the code

1)The file for facial landmark detection is “shape_predictor_68_face_ landmarks.dat”. This can be found in internet.
2)The FINAL2 folder contains the dataset of eye images of up, down, right, centre, left, open and close. You can download and unzip  to use. The dataset has been created using some images from Eyechimera dataset  and Closed Eyes in the Wild(CEW) dataset which is available in internet and also some which is created manually by ourselves.
3)3class model creation contains the code for model creation for right, centre and left classification. You can use the dataset in FINAL2 for training.
4)5class model creation contains the code for model creation for right, centre, up, down and left classification. You can use the dataset in FINAL2 for training.
5)Blink model creation contains the code for model creation for open and close eye classification. You can use the dataset in FINAL2 for training.
6)After creating the model you can run the code for 3 class classification inorder to classify to left right centre and blink in video.
7)After creating the model you can run the code for 5 class classification inorder to classify to left right centre up down and blink in video.
8)Create empty folders Framesleft, Framesright,Framesleft1, Framesright1 for smooth execution of code.These are used to store some images in between the process.


