
```markdown
# CNN Image Classifier - Cats vs Dogs (PyTorch)

This project is a **Deep Learning Image Classifier** using **Convolutional Neural Networks (CNNs)** built with **PyTorch**.  
The model classifies images of cats and dogs using **data augmentation** to improve generalization.

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn (for train/test split)

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 🏗 Workflow

1. **Data Preparation**

   * Organize the dataset into `train` and `test` folders.
   * Each class (`cats`, `dogs`) should have its own subfolder.

2. **Data Augmentation**

   * Apply random rotations and horizontal flips.
   * Increases dataset variability and helps the model generalize better.

3. **CNN Model Architecture**

   * 3 Convolutional layers with ReLU activation
   * MaxPooling layers after convolutions
   * Fully connected layers
   * Output layer with **2 neurons** for binary classification

4. **Training**

   * Optimizer: Adam
   * Loss: CrossEntropyLoss
   * Metrics: Accuracy
   * Epochs: 10 (can be increased for higher accuracy)
   * Optional: GPU acceleration if available

5. **Evaluation**

   * Compute test accuracy after training.
   * Monitor training loss during epochs.

6. **Model Saving**

   * Save trained model with `torch.save(model.state_dict(), "cnn_cats_vs_dogs.pth")`
   * Can be loaded later for inference or deployment.

---

## ⚡ How to Run

```bash
python main.py
```

The script will:

* Load images from the dataset
* Apply augmentation
* Train the CNN
* Display training loss
* Evaluate accuracy on the test set

---

## 🖼 Notes

* Recommended image size: 128x128 pixels.
* For large datasets, adjust `batch_size` or use GPU acceleration.
* Ensure all images are valid (corrupted images may break training).

---

## 🔗 References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)



