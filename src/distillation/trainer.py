import torch
import torch.nn as nn
import tqdm as tqdm


class DistillationTrainer:

    def __init__(self, student, teacher, optimizer, device,
                 use_gt=True,
                 use_teacher_hard=False,
                 use_teacher_soft=False
                ):

        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device

        self.use_gt = use_gt
        self.use_teacher_hard = use_teacher_hard
        self.use_teacher_soft = use_teacher_soft

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        if self.teacher is not None:
            self.teacher.eval()

    def train(self, train_loader, epochs):
        print("Distillation Started!")
        for epoch in range(epochs):
            self.student.train()
            # Progress bar for batches
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                x_org = x
                x = x.view(x.size(0), -1)

                student_logits = self.student(x)

                loss = 0

                # ---- Ground Truth Loss ----
                if self.use_gt:
                    loss_gt = self.ce(student_logits, y)
                    loss += loss_gt

                # ---- Teacher Forward ----
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_logits = self.teacher(x_org)

                # ---- Hard Teacher Loss ----
                if self.use_teacher_hard:
                    teacher_preds = torch.argmax(teacher_logits, dim=1)
                    loss_hard = self.ce(student_logits, teacher_preds)
                    loss += loss_hard

                # ---- Soft Teacher Loss ----
                if self.use_teacher_soft:
                    loss_soft = self.mse(student_logits, teacher_logits)
                    loss += loss_soft

                # ---- Backprop ----
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()