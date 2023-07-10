import time
from sklearn.metrics import roc_auc_score, roc_curve

import torch


def train_val_class(epoch, dataloader, model, optimizer, mixed_precision=True, device='cuda', train=True):
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
	for idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
		with torch.set_grad_enabled(train):
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			labels = labels.to(device)
			epoch_samples += len(input_ids)
		#   optimizer.zero_grad()
			with torch.cuda.amp.autocast(mixed_precision):
				outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
				loss = outputs.loss
				logits = outputs.logits
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
				pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
				lab.extend(labels.cpu().numpy())
				if train:
					msg = f"Epoch: {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
				else:
					msg = f'Epoch {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
					print(msg, end= '\r')
	
	roc_auc = roc_auc_score(lab, pred)
	false_positive_rate, true_positive_rate, thresolds = roc_curve(lab, pred)
	# plot_confusion_matrix(pred, lab, [0, 1])
	if train: stage='train'
	else: stage='validation'
	msg = f'{stage} Loss: {running_loss/epoch_samples:.4f} \n {stage} ROC_AUC: {roc_auc:.4f}'
	print(msg)
	return running_loss/epoch_samples, roc_auc, false_positive_rate, true_positive_rate, thresolds