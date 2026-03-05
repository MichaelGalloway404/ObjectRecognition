# ObjectRecognition
# Object Classification Demo Using ResNet-9

This project is a demo of a Convolutional Neural Network (CNN) model trained and presented for a research course at **Eastern Oregon University (Fall 2024)**. The model leverages the **ResNet-9 architecture** to classify images captured via a webcam.

## Features

* **Object Classification:** Distinguishes between planes, cars, birds, cats, deer, dogs, frogs, horses, humans, or ships.
* **Dataset:** Trained primarily on the CIFAR-10 dataset, with additional images to improve recognition of human faces.
* **Live Demo:** Users can capture images using a webcam by pressing the spacebar. The system will classify objects in real-time, including physical objects or printed images.
* **Flexible Training:** Includes a script to retrain the model on custom datasets and generate a `.pth` file for use in the demo.

## Project Structure

* **`model_trainer.py`**
  Trains the ResNet-9 CNN on images located in the `cifar10` folder and any additional images. Outputs a `.pth` model file for later use.

* **`trained_ObjAnalysis_demo.py`**
  Runs the live object classification demo. Captures images from a webcam, processes them, and predicts the object category using the trained ResNet-9 model.

## How It Works

1. **Model Training:**
   Run `model_trainer.py` to train the ResNet-9 model on your dataset. A trained model file (e.g., `model_202506241202.pth`) will be generated.

2. **Object Analysis Demo:**

   * Run `trained_ObjAnalysis_demo.py`.
   * A window opens showing the webcam feed with a green square overlay.
   * Place the object or image within the square.
   * Press **SPACEBAR** to capture and classify the object.
   * Press **Q** to exit the demo.

3. **Processing:**

   * Captured images are cropped, resized to 32x32 pixels, normalized, and passed to the model.
   * The model outputs the predicted class: Plane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Person, Ship, or Unknown.

## Dependencies

* Python 3.8+
* `torch` and `torchvision`
* `opencv-python`
* `Pillow`
* `matplotlib`

Install dependencies using:

```bash
pip install torch torchvision opencv-python pillow matplotlib
```

## Author

**Michael Galloway**
Presented for Research Course, Eastern Oregon University, Fall 2024
