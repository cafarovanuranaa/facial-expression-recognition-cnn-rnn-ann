# 📘 FACE EXPRESSION RECOGNITION PIPELINE 

**Introduction:**   
This project loads images from [Google Drive](https://drive.google.com/drive/folders/1rcwcbJQZ8VHEeGeM4-16hqE2zNG9lrfZ?usp=drive_link) to recognize facial expressions using a CNN+RNN+ANN hybrid model. Each step shows exactly what is done.

---

## 1. Training Data Generator
- Rescales pixel values (`1./255`) to normalize images.  
- Applies data augmentation: shear, zoom, horizontal flip.  
- Loads images from training folder.  
- Resizes images to 64×64.  
- Groups images into batches of 32.  
- Assigns categorical labels automatically based on folder names.

---

## 2. Test Data Generator
- Only rescaling (no augmentation).  
- Loads images from test folder.  
- Resizes to 64×64, batch size 32.  
- Ensures clean evaluation with categorical labels.

---

## 3. Model Creation (Optuna)
- **CNN layers**: extract spatial patterns using Conv2D, BatchNormalization, MaxPooling, and SpatialDropout.  
- **Flatten + Reshape**: converts CNN output into sequences for RNN.  
- **RNN layers**: LSTM or GRU, chosen by Optuna, process sequences to capture temporal relationships.  
- **Dense (ANN) layers**: combine high-level features, apply Dropout and BatchNormalization.  
- **Output layer**: softmax for 7 facial expression classes.  
- Optimizer and learning rate are selected by Optuna.

---

## 4. Objective Function
- Defines what Optuna optimizes: minimum validation loss.  
- Trains the model using the trial's hyperparameters.  
- Applies EarlyStopping: stops if validation loss does not improve.  
- Applies ReduceLROnPlateau: reduces learning rate when progress stalls.  
- Returns best validation loss for the trial.

---

## 5. Running Optuna Study
- Uses **TPE sampler** for efficient search.  
- Runs multiple trials to find the best hyperparameters.  
- After study, selects best hyperparameters.

---

## 6. Final Model
- Rebuilds model using best parameters from Optuna.  
- Trains on full training set with EarlyStopping and ReduceLROnPlateau.  
- Epochs up to 50, batch size from best params.

---

## 7. Evaluation
- Evaluate on training set to check learning.  
- Evaluate on test set to check generalization.  
- Prints loss and accuracy for both sets.

---

## 8. Image Display
- Converts images to HTML-friendly format using base64.  
- Allows images to be displayed in a table alongside predictions.

---

## 9. Single Image Prediction
- Loops through folder with new images.  
- Loads, resizes to 64×64, and normalizes images.  
- Predicts class using the trained model.  
- Converts predicted index to class label.  
- Stores both image and prediction in a results list.

---

## 10. Display Results
- Converts results list to DataFrame.  
- Displays as HTML table with images and predictions for clear visualization.
