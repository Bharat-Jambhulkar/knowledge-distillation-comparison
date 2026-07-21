# Experimental Comparison of Knowledge Distillation Techniques in Neural Networks

This project implements and compares different knowledge distillation approaches on the Fashion MNIST dataset. Knowledge distillation is a technique in which a smaller (also called a student) model is trained using a larger, well-trained (also called a teacher) model to achieve improved performance with reduced model size and complexity.

This project explores and compares three distinct training strategies:

- **Black-Box Distillation**: The student trained from teacher predictions.
- **White-Box Distillation**: The student trained from soft targets from the teacher model.
- **Supervised Learning**: The student is trained with ground truth labels only.

## 📁 Repository Structure

```
knowledge-distillation-comparison/
├── main.py                          
├── requirements.txt                 
├── data/                            
│   ├── fashion-mnist_train.csv
│   ├── fashion-mnist_test.csv
│   ├── fashion-mnist_teacher_train.csv
│   └── fashion-mnist_distillation_train.csv
├── src/                             
│   ├── models/                      # Model architectures
│   │   ├── teacher.py               
│   │   ├── student.py               
│   │   └── trained-teacher-model/
│   │       ├── cnn_model.pth        # Trained teacher weights
│   │       └── trained-teacher-model-checkpoint.pth
│   ├── training/                    # Training utilities
│   │    └──data_loader.py                          
│   ├── distillation/                # Distillation implementation
│   │   └── trainer.py               
│   └── evaluation/               
│       └── comparison.py            # Model evaluation and comparison
│
├──LICENSE 
└──README.md                        
```

## 📊 Results

The following table summarizes the performance of different approaches:

| Method                          | Accuracy |
|---------------------------------|----------|
| Only Black Box                  | 0.8785   |
| Only White Box                  | 0.8756   |
| Black Box + Supervised          | 0.8891   |
| White Box + Supervised          | 0.8759   |
| Only Supervised                 | 0.8867   |


To evaluate the impact of distillation dataset size on model performance, a controlled experiment comparing the three approaches: (i) supervised learning, (ii) supervised learning combined with white-box distillation and (iii) supervised learning combined with black-box distillation was conducted. The results, shown in the figure below, indicate that the combined approach consistently achieves higher accuracy across varying sample sizes. These findings suggest that incorporating distillation provides benefits, even when ground-truth labels are available.

![Accuracy Plot](/data/acc-sample-vary.png)

## Run the Project Locally

### Clone the Repository

```bash
git clone https://github.com/Bharat-Jambhulkar/knowledge-distillation-comparison.git
cd knowledge-distillation-comparison
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run All Experiments

Execute all three distillation approaches to compare their performance:

```bash
python main.py
```

### Run Specific Experiment

To run a specific experiment by name modify the last line in `main.py` to specify the experiment:

```python
if __name__ == "__main__":
    main("White-Box Distillation", num_samples=30000)
```

Available experiment names:
- `Black-Box Distillation`
- `White-Box Distillation`
- `Supervised Learning`

### Python Version

- Python 3.7 or higher
