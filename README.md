# CS636 Assignmnent 4 GCN

This repository is forked from https://github.com/the-data-lab/Deep-Education.

## Usage

To test the file: <br/>
```
python3 kernel/GCN_pubmed.py
```

The completed code is in /kernel. The files in dl_code_python are the original skeleton codes.

## Sample Output
Single-thread GCN: <br/>
```
Epoch 190 | Train_Loss: 0.0210
Epoch 191 | Train_Loss: 0.0210
Epoch 192 | Train_Loss: 0.0209
Epoch 193 | Train_Loss: 0.0208
Epoch 194 | Train_Loss: 0.0208
Epoch 195 | Train_Loss: 0.0207
Epoch 196 | Train_Loss: 0.0206
Epoch 197 | Train_Loss: 0.0206
Epoch 198 | Train_Loss: 0.0205
Epoch 199 | Train_Loss: 0.0205
the time of graphpy is: 0:00:05.507150
Epoch 199 | Test_accuracy: 0.7280
```
Multi-thread GCN: <br/>
```
Epoch 190 | Train_Loss: 0.0207
Epoch 191 | Train_Loss: 0.0203
Epoch 192 | Train_Loss: 0.0202
Epoch 193 | Train_Loss: 0.0201
Epoch 194 | Train_Loss: 0.0200
Epoch 195 | Train_Loss: 0.0200
Epoch 196 | Train_Loss: 0.0200
Epoch 197 | Train_Loss: 0.0198
Epoch 198 | Train_Loss: 0.0198
Epoch 199 | Train_Loss: 0.0197
the time of graphpy is: 0:00:03.999862
Epoch 199 | Test_accuracy: 0.7300
```
