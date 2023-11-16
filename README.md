# LeNet
LeNet CNN for homework 7

running and installing code-

preconditions:pip,python,pandas shold be installed on system

step1:create a folder A with all code files and text files

step2:make a virtual environment on the system inside the folder A-call this VE say CNN using venv

```python -m venv CNN```

this creates a folder CNN with subfolders Scripts,venv etc-Scripts contains all the installed libraries we need(but we have to install pytorch and tqdm)

step3:activate VE

go into folder A

```CNN/Scripts/activate```

step4:install pytorch and tqdm(we are working now fully from now on in the virtual environment

installing and upgrading pip-go into folder A then type this

```python.exe -m pip install --upgrade pip```

```pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0```

go to CNN/Scripts

```pip install tqdm```

now check existence of tqdm using

```pip show tqdm```

```pip list```





