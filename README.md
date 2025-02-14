# CS636 Assignmnent 4 GCN & Assignemnt 5 Multi-Thread GCN

This repository is forked from https://github.com/the-data-lab/Deep-Education.

## Usage

There should be no need to compile the code. To test single-thread GCN: <br/>
```
python3 kernel/GCN_pubmed.py
```
To test multi-thread GCN: <br/>
```
python3 kernel_mt/GCN_pubmed.py
```

The completed code is in /kernel and /kernel_mt. The files in dl_code_python are the original skeleton codes.

## Compilation Guide
To compile: <br/>
```
cd kernel or cd kernel_mt
cmake .
make
```
If you encounter the following error
```
The source directory

    /home/zhudi/bin/CS636_Big_Data/as5/Deep-Education/kernel/pybind11

  does not contain a CMakeLists.txt file.

```
Please do
```
git rm --cached pybind11
rm -r pybind11
git submodule add https://github.com/pybind/pybind11.git pybind11
```
and then compile.

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
