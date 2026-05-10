import torch
import torch.optim as optim

from src.models.student import StudentANN
from src.models.teacher import TeacherCNN
from src.training.data_loader import get_dataloaders
from src.distillation.trainer import DistillationTrainer
from src.evaluation.comparison import evaluate
from torch.utils.data import Subset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)


def build_teacher():
    teacher = TeacherCNN(input_features=1).to(device)
    teacher.load_state_dict(torch.load(r"src\models\trained-teacher-model\cnn_model.pth"))
    print("Teacher Model Loaded!")
    return teacher


def run_experiment(name, config, train_loader, test_loader):
    print(f"\nRunning: {name}")

    student = StudentANN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=6.7790e-5)

    teacher = build_teacher() if config["use_teacher"] else None

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        use_gt=config["use_gt"],
        use_teacher_hard=config["teacher_hard"],
        use_teacher_soft=config["teacher_soft"],
    )

    trainer.train(train_loader=train_loader, epochs=config["epochs"])

    acc = evaluate(model=student, test_loader=test_loader, device=device)

    print(f"{name} Accuracy: {acc:.4f}")


def main(experiment_name=None, num_samples=None):
    train_loader, test_loader = get_dataloaders(batch_size=32)
    
    if num_samples is not None:
        dataset = train_loader.dataset

        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)

        train_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=True
        )

    experiments = {
        "Black-Box Distillation": {
            "use_gt": False,
            "teacher_hard": True,
            "teacher_soft": False,
            "use_teacher": True,
            "epochs": 24,
        },
        "White-Box Distillation": {
            "use_gt": False,
            "teacher_hard": False,
            "teacher_soft": True,
            "use_teacher": True,
            "epochs": 24,
        },
        "Supervised Learning": {
            "use_gt": True,
            "teacher_hard": False,
            "teacher_soft": False,
            "use_teacher": False,
            "epochs": 24,
        },
    }

    if experiment_name:
        config = experiments[experiment_name]
        run_experiment(experiment_name, config, train_loader, test_loader)
    else:
        # run all
        for name, config in experiments.items():
            run_experiment(name, config, train_loader, test_loader)


if __name__ == "__main__":
    main("Black-Box Distillation", num_samples=30000)
    main("White-Box Distillation", num_samples=30000)
    main("Supervised Learning", num_samples=30000)