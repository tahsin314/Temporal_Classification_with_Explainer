import time
from sklearn.metrics import accuracy_score

import torch
import wandb


def train_val_class(epoch, dataloader, class_names, model, 
					criterion, optimizer, mixed_precision=True, 
					device='cuda', train=True):
	t1 = time.time()
	running_loss = 0
	epoch_samples = 0
	pred = []
	lab = []
	scaler = torch.cuda.amp.GradScaler()
	if train:
		model.train()
		print("Initiating train phase ...")
	else:
		model.eval()
		print("Initiating val phase ...")
	for idx, (data, labels) in enumerate(dataloader):
		with torch.set_grad_enabled(train):
			data = data.to(device).to(dtype=torch.float32)
			labels = labels.to(device).to(dtype=torch.long)

			epoch_samples += len(data)
		#   optimizer.zero_grad()
			with torch.cuda.amp.autocast(mixed_precision):
				outputs = model(data)
				loss = criterion(outputs, labels)
				
				running_loss += loss.item()

				if train:
					if mixed_precision:
						scaler.scale(loss).backward()
						scaler.step(optimizer) 
						scaler.update() 
						optimizer.zero_grad()
					else:
						loss.backward()
						optimizer.step()
						optimizer.zero_grad()

				elapsed = int(time.time() - t1)
				eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
				pred.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
				lab.extend(labels.cpu().numpy())
				if train:
					msg = f"Epoch: {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
					wandb.log({"Train Loss": running_loss/epoch_samples, "Epoch": epoch})
				else:
					msg = f'Epoch {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
					wandb.log({"Validation Loss": running_loss/epoch_samples, "Epoch": epoch})
				print(msg, end= '\r')
	
	accuracy = accuracy_score(lab, pred)

	# false_positive_rate, true_positive_rate, thresolds = roc_curve(lab, pred)
	# plot_confusion_matrix(pred, lab, [0, 1])
	if train: 
		stage='train'
		wandb.log({"Train Accuracy": accuracy, "Epoch": epoch})
	else: 
		stage='validation'
		wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=lab, preds=pred,
                        class_names=class_names)})
		wandb.log({"Validation Accuracy": accuracy, "Epoch": epoch})
	msg = f'{stage} Loss: {running_loss/epoch_samples:.4f} \n {stage} ROC_AUC: {accuracy:.4f}'
	print(msg)
	return running_loss/epoch_samples, accuracy